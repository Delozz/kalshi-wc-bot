"""Tests for the execution layer (order construction, demo guard, portfolio math)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from execution import order_manager, portfolio
from schemas import Signal


def _signal(*, price: float = 0.50, bet_cents: int = 5000) -> Signal:
    return Signal(
        match_id="FRA_MAR_2026-06-20",
        market_ticker="KXWC26-FRA",
        side="YES",
        model_prob=0.62,
        market_implied=price,
        edge=0.12,
        kelly_fraction=0.05,
        bet_size_cents=bet_cents,
        generated_at=datetime.now(timezone.utc),
    )


def test_order_from_signal_contract_count() -> None:
    # $50 bet at 50c per contract -> 100 contracts.
    request = order_manager.order_from_signal(_signal(price=0.50, bet_cents=5000))
    assert request is not None
    assert request.count == 100
    assert request.limit_price_cents == 50
    assert request.side == "bid"  # V2 API: YES side = "bid"


def test_order_from_signal_below_one_contract() -> None:
    # $0.30 bet at 50c per contract -> 0 contracts -> no order.
    assert order_manager.order_from_signal(_signal(price=0.50, bet_cents=30)) is None


def test_order_from_signal_rejects_bad_price() -> None:
    assert order_manager.order_from_signal(_signal(price=0.0)) is None
    assert order_manager.order_from_signal(_signal(price=1.0)) is None


def test_dry_run_does_not_send() -> None:
    result = asyncio.run(order_manager.place_order(_signal(), dry_run=True))
    assert result is not None
    assert result["status"] == "dry_run"
    assert result["order_id"] is None


def test_place_order_extracts_order_id(monkeypatch) -> None:
    # Opt in to the prod-order path so the test is hermetic regardless of the ambient
    # KALSHI_ENV in .env (it exercises a real placement, not the L8 demo-first guard).
    monkeypatch.setenv("KALSHI_ALLOW_PROD_ORDERS", "1")

    async def fake_create_order(**_kwargs):
        return {"order_id": "abc123", "fill_count": "0.00", "remaining_count": "10.00"}

    monkeypatch.setattr(order_manager.kalshi, "create_order", fake_create_order)
    result = asyncio.run(order_manager.place_order(_signal(), dry_run=False))
    assert result is not None
    assert result["status"] == "placed"
    assert result["order_id"] == "abc123"


def test_portfolio_exposure_and_peak() -> None:
    state = portfolio.PortfolioState(bankroll_cents=20000, peak_bankroll_cents=20000)
    state.positions = [
        portfolio.Position("A", "yes", count=100, avg_price_cents=50),
        portfolio.Position("B", "yes", count=20, avg_price_cents=70),
    ]
    assert state.exposure_cents == 100 * 50 + 20 * 70
    assert state.open_count == 2

    state.update_bankroll(25000)
    assert state.peak_bankroll_cents == 25000
    state.update_bankroll(22000)
    assert state.peak_bankroll_cents == 25000  # peak ratchets, does not fall
