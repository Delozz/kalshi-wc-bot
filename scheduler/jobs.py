"""Scheduler (scheduler/jobs.py) — APScheduler jobs for the live tournament.

Wires the working pieces (market-data refresh, portfolio sync, dry-run order flow) on
the cadence from PRD section 12. Every job is wrapped so one failure never stops the
scheduler (L9). Signal generation loads the latest model artifact; mapping a Kalshi
market to a fixture's feature vector needs the API-Football fixtures client (T-24h),
which is the remaining live integration — the job logs that gap rather than guessing.

Run: ``python -m scheduler.jobs``  (demo env until validated, L8).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pickle
from pathlib import Path
from typing import Any

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import ARTIFACTS_DIR, configure_logging, ensure_dirs, settings
from execution import portfolio
from ingestion import kalshi, odds_api

logger = logging.getLogger(__name__)

# Shared mutable context passed to jobs (portfolio state, loaded model artifact).
CONTEXT: dict[str, Any] = {"artifact": None, "portfolio": None}


def load_latest_artifact() -> dict[str, Any] | None:
    """Load the most recent trained model artifact, or None if none exists."""
    artifacts = sorted(Path(ARTIFACTS_DIR).glob("model_*.pkl"))
    if not artifacts:
        logger.warning("No model artifact found; run `python -m model.train` first")
        return None
    with open(artifacts[-1], "rb") as handle:
        artifact = pickle.load(handle)
    logger.info("Loaded model artifact %s", artifacts[-1].name)
    return artifact


def _safe(job_name: str, fn: Any) -> None:
    """Run a job body, logging and swallowing any exception (L9)."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — a scheduler job must never crash the loop
        logger.exception("Job %s failed: %s", job_name, exc)


def job_refresh_market_data() -> None:
    """Pull Kalshi WC markets and The Odds API snapshot (cached)."""

    def _run() -> None:
        markets = asyncio.run(kalshi.get_markets())
        odds = asyncio.run(odds_api.fetch_odds())
        logger.info(
            "Refreshed market data: %d Kalshi markets, %d odds events",
            len(markets),
            len(odds),
        )

    _safe("refresh_market_data", _run)


def job_sync_portfolio() -> None:
    """Sync bankroll and positions from Kalshi into the shared context."""

    def _run() -> None:
        state = asyncio.run(
            portfolio.sync_from_kalshi(
                fallback_bankroll_cents=settings.initial_bankroll_cents
            )
        )
        CONTEXT["portfolio"] = state

    _safe("sync_portfolio", _run)


def job_generate_signals() -> None:
    """Generate dry-run signals from fixtures + model + live Kalshi prices."""

    def _run() -> None:
        from strategy import signal_gen

        dry_run = bool(CONTEXT.get("dry_run_orders", True))
        signals = asyncio.run(signal_gen.run_live(dry_run=dry_run))
        logger.info(
            "generate_signals produced %d signal(s) (dry_run=%s)", len(signals), dry_run
        )

    _safe("generate_signals", _run)


def job_settle_positions() -> None:
    """Settle finished fixtures' orders and post realized P&L to the ledger."""

    def _run() -> None:
        asyncio.run(_settle_finished())

    _safe("settle_positions", _run)


async def _settle_finished() -> None:
    from data.db import connect, init_db, record_bankroll
    from execution import portfolio, settlement
    from ingestion import api_football

    raw = await api_football.fetch_fixtures()
    done = api_football.finished(api_football.parse_fixtures(raw))
    if not done:
        logger.info("No finished fixtures to settle")
        return
    init_db()
    with connect() as conn:
        for fixture in done:
            result = api_football.outcome(fixture)
            if result is None:
                continue
            pnl = settlement.settle_fixture(conn, fixture.fixture_id, result)
            logger.info(
                "Settled fixture %s (%s): pnl=%dc", fixture.fixture_id, result, pnl
            )
    # Bankroll is authoritative from Kalshi (settlement cash is already reflected in the
    # account balance); re-sync so the ledger matches reality rather than drifting on an
    # additive estimate. A fallback balance is labelled so it never pollutes the peak.
    state = await portfolio.sync_from_kalshi(
        fallback_bankroll_cents=settings.initial_bankroll_cents
    )
    event = "sync_fallback" if state.balance_is_fallback else "sync"
    with connect() as conn:
        record_bankroll(conn, state.bankroll_cents, event)


def job_update_bankroll() -> None:
    """Sync bankroll from Kalshi, record it, and alarm if the stop-loss is breached."""

    def _run() -> None:
        asyncio.run(_update_bankroll())

    _safe("update_bankroll", _run)


async def _update_bankroll() -> None:
    from data.db import connect, init_db, real_peak_bankroll, record_bankroll
    from execution import portfolio
    from strategy.risk import stop_loss_triggered

    state = await portfolio.sync_from_kalshi(
        fallback_bankroll_cents=settings.initial_bankroll_cents
    )
    init_db()
    event = "sync_fallback" if state.balance_is_fallback else "sync"
    with connect() as conn:
        # Ratchet against prior real syncs, then record this balance under the event label
        # that keeps fallbacks out of the high-water mark.
        portfolio.ratchet_peak(state, real_peak_bankroll(conn))
        record_bankroll(conn, state.bankroll_cents, event)
    if stop_loss_triggered(
        state.bankroll_cents / 100.0, state.peak_bankroll_cents / 100.0
    ):
        logger.error(
            "STOP-LOSS BREACHED: bankroll %dc vs peak %dc — halt betting",
            state.bankroll_cents,
            state.peak_bankroll_cents,
        )


def build_scheduler() -> BlockingScheduler:
    """Build (but do not start) the scheduler with all jobs registered."""
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        job_refresh_market_data, CronTrigger(hour="*/6"), id="refresh_odds"
    )
    scheduler.add_job(
        job_sync_portfolio, CronTrigger(minute="*/30"), id="sync_portfolio"
    )
    scheduler.add_job(
        job_generate_signals, CronTrigger(hour="*/3"), id="generate_signals"
    )
    scheduler.add_job(
        job_settle_positions, CronTrigger(hour="*/2"), id="settle_positions"
    )
    scheduler.add_job(
        job_update_bankroll, CronTrigger(minute="*/30"), id="update_bankroll"
    )
    return scheduler


def _render_dashboard() -> None:
    """Render the dashboard for the current portfolio + recent signals (best effort)."""
    try:
        from dashboard.app import _position_theses, _recent_signals, render
        from execution.portfolio import PortfolioState

        state = CONTEXT.get("portfolio")
        if not isinstance(state, PortfolioState):
            state = PortfolioState(
                bankroll_cents=settings.initial_bankroll_cents,
                peak_bankroll_cents=settings.initial_bankroll_cents,
            )
        render(
            state,
            _recent_signals(),
            position_theses=_position_theses(state.positions),
        )
    except Exception as exc:  # noqa: BLE001 — dashboard is non-critical
        logger.warning("Dashboard render failed: %s", exc)


def run_cycle(*, dry_run_orders: bool = True) -> None:
    """Run one full pass of every job, then render the dashboard (demo paper run).

    With ``dry_run_orders=False`` and ``KALSHI_ENV=demo`` this places real demo orders;
    prod still requires KALSHI_ALLOW_PROD_ORDERS=1 (L8).
    """
    CONTEXT["dry_run_orders"] = dry_run_orders
    if CONTEXT["artifact"] is None:
        CONTEXT["artifact"] = load_latest_artifact()
    job_refresh_market_data()
    job_sync_portfolio()
    job_generate_signals()
    job_settle_positions()
    job_update_bankroll()
    _render_dashboard()


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi WC bot scheduler.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one full cycle and exit (demo paper run).",
    )
    parser.add_argument(
        "--live-orders",
        action="store_true",
        help="Place real orders instead of dry-run (demo env unless prod opt-in, L8).",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()
    CONTEXT["artifact"] = load_latest_artifact()

    if args.once:
        logger.info(
            "Running one cycle (env=%s, live_orders=%s)",
            settings.kalshi_env,
            args.live_orders,
        )
        run_cycle(dry_run_orders=not args.live_orders)
        return

    # Persistent scheduler: thread the order mode into CONTEXT so job_generate_signals
    # honours --live-orders. Without this it falls back to dry_run=True and an automated
    # loop would never place real orders. Prod still requires KALSHI_ALLOW_PROD_ORDERS=1
    # at the order-manager layer (L8), and the stop-loss still halts betting via the risk
    # check inside signal generation — both hold even when running unattended.
    CONTEXT["dry_run_orders"] = not args.live_orders
    scheduler = build_scheduler()
    logger.info(
        "Starting scheduler (env=%s, live_orders=%s) with jobs: %s",
        settings.kalshi_env,
        args.live_orders,
        [job.id for job in scheduler.get_jobs()],
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
