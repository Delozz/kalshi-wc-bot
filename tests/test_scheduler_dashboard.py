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


class _FakeScheduler:
    """Stand-in for the blocking scheduler: exits start() immediately."""

    def get_jobs(self) -> list:
        return []

    def start(self) -> None:
        raise SystemExit  # main() catches this and returns cleanly


def _run_main(monkeypatch, argv: list[str]) -> bool:
    """Run scheduler main() with a fake (non-blocking) scheduler; return the order mode."""
    import sys

    import scheduler.jobs as jobs

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(jobs, "load_latest_artifact", lambda: None)
    monkeypatch.setattr(jobs, "build_scheduler", lambda: _FakeScheduler())
    # These tests exercise order-mode threading, not the single-instance lock; stub it as
    # acquired so a real bind (and its process-lifetime socket) doesn't leak across tests.
    monkeypatch.setattr(jobs, "acquire_single_instance_lock", lambda: True)
    jobs.CONTEXT["dry_run_orders"] = None  # ensure main() is what sets it
    jobs.main()
    return jobs.CONTEXT["dry_run_orders"]


def test_persistent_scheduler_threads_live_orders(monkeypatch) -> None:
    # The blocking scheduler must honour --live-orders. This is the money-critical gap:
    # without threading the flag into CONTEXT, job_generate_signals defaults to dry_run.
    assert _run_main(monkeypatch, ["jobs", "--live-orders"]) is False  # live = not dry


def test_persistent_scheduler_defaults_to_dry_run(monkeypatch) -> None:
    # No --live-orders: the automated loop must stay in dry-run (never place real orders).
    assert _run_main(monkeypatch, ["jobs"]) is True


def test_single_instance_lock_blocks_second_holder(monkeypatch) -> None:
    # The first bind acquires the lock; a second must be refused (two daemons = double
    # orders). Patch the port so the test never collides with a real running scheduler.
    import scheduler.jobs as jobs

    monkeypatch.setattr(jobs, "_SINGLE_INSTANCE_PORT", 49519)
    monkeypatch.setattr(jobs, "_instance_lock", None)
    first = jobs.acquire_single_instance_lock()
    try:
        assert first is True
        assert jobs.acquire_single_instance_lock() is False  # port already held
    finally:
        if jobs._instance_lock is not None:
            jobs._instance_lock.close()
            jobs._instance_lock = None


def test_second_scheduler_instance_exits(monkeypatch) -> None:
    # main() must refuse to start a persistent loop when the lock is already held.
    import sys

    import pytest

    import scheduler.jobs as jobs

    monkeypatch.setattr(sys, "argv", ["jobs", "--live-orders"])
    monkeypatch.setattr(jobs, "load_latest_artifact", lambda: None)
    monkeypatch.setattr(jobs, "build_scheduler", lambda: _FakeScheduler())
    monkeypatch.setattr(jobs, "acquire_single_instance_lock", lambda: False)
    with pytest.raises(SystemExit):
        jobs.main()
