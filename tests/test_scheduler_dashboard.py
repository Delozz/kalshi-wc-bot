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
