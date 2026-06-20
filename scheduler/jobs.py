"""Scheduler (scheduler/jobs.py) — APScheduler jobs for the live tournament.

Wires the working pieces (market-data refresh, portfolio sync, dry-run order flow) on
the cadence from PRD section 12. Every job is wrapped so one failure never stops the
scheduler (L9). Signal generation loads the latest model artifact; mapping a Kalshi
market to a fixture's feature vector needs the API-Football fixtures client (T-24h),
which is the remaining live integration — the job logs that gap rather than guessing.

Run: ``python -m scheduler.jobs``  (demo env until validated, L8).
"""

from __future__ import annotations

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
    """Score upcoming fixtures against Kalshi prices (pending fixtures integration)."""

    def _run() -> None:
        if CONTEXT["artifact"] is None:
            CONTEXT["artifact"] = load_latest_artifact()
        markets = asyncio.run(kalshi.get_markets())
        logger.info(
            "Signal generation: %d markets seen. Fixture->feature mapping requires the "
            "API-Football fixtures client (remaining live integration).",
            len(markets),
        )

    _safe("generate_signals", _run)


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
    return scheduler


def main() -> None:
    configure_logging()
    ensure_dirs()
    CONTEXT["artifact"] = load_latest_artifact()
    scheduler = build_scheduler()
    logger.info(
        "Starting scheduler (env=%s) with jobs: %s",
        settings.kalshi_env,
        [job.id for job in scheduler.get_jobs()],
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
