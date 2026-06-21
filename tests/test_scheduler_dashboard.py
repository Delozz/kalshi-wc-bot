"""Smoke tests for the scheduler wiring and the dashboard renderer."""

from __future__ import annotations

import io

from rich.console import Console

from dashboard.app import render
from execution.portfolio import PortfolioState, Position
from scheduler.jobs import build_scheduler


def test_scheduler_registers_all_jobs() -> None:
    scheduler = build_scheduler()
    job_ids = {job.id for job in scheduler.get_jobs()}
    assert job_ids == {
        "refresh_odds",
        "sync_portfolio",
        "generate_signals",
        "settle_positions",
        "update_bankroll",
    }


def test_dashboard_renders_without_error() -> None:
    state = PortfolioState(
        bankroll_cents=20000,
        peak_bankroll_cents=20500,
        positions=[Position("KXWC26-FRA", "yes", 100, 50)],
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100)
    render(state, [], console=console)
    output = buffer.getvalue()
    assert "KALSHI WC BOT" in output
    assert "KXWC26-FRA" in output


def test_dashboard_renders_empty_portfolio() -> None:
    state = PortfolioState(bankroll_cents=0, peak_bankroll_cents=0)
    buffer = io.StringIO()
    console = Console(file=buffer, width=100)
    render(state, [], console=console)
    assert "OPEN POSITIONS" in buffer.getvalue()


def test_run_cycle_invokes_all_jobs_in_order(monkeypatch) -> None:
    import scheduler.jobs as jobs

    calls: list[str] = []
    monkeypatch.setattr(jobs, "load_latest_artifact", lambda: None)
    for name in (
        "job_refresh_market_data",
        "job_sync_portfolio",
        "job_generate_signals",
        "job_settle_positions",
        "job_update_bankroll",
        "_render_dashboard",
    ):
        monkeypatch.setattr(jobs, name, (lambda n: lambda: calls.append(n))(name))

    jobs.run_cycle(dry_run_orders=False)

    assert calls == [
        "job_refresh_market_data",
        "job_sync_portfolio",
        "job_generate_signals",
        "job_settle_positions",
        "job_update_bankroll",
        "_render_dashboard",
    ]
    assert jobs.CONTEXT["dry_run_orders"] is False
