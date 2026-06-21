"""Tests for settlement P&L and the bankroll ledger (execution/settlement.py)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data import db
from execution import settlement
from schemas import Order, Signal


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "test.sqlite"
    db.init_db(path)
    connection = db.connect(path)
    yield connection
    connection.close()


def _signal(match_id: str = "Brazil_Serbia_999") -> Signal:
    return Signal(
        match_id=match_id,
        market_ticker="KXWC26-BRA",
        side="YES",
        model_prob=0.60,
        market_implied=0.50,
        edge=0.10,
        kelly_fraction=0.05,
        bet_size_cents=1000,
        generated_at=datetime.now(timezone.utc),
    )


def _order(
    signal_id: int, *, order_id: str = "ord1", filled: float | None = 0.50
) -> Order:
    return Order(
        id=order_id,
        signal_id=signal_id,
        status="filled",
        limit_price=0.50,
        contracts=10,
        filled_price=filled,
        placed_at=datetime.now(timezone.utc),
        settled_at=None,
        pnl_cents=None,
    )


def test_compute_pnl_win_and_loss() -> None:
    assert settlement.compute_pnl_cents(60, 10, won=True) == 10 * 40
    assert settlement.compute_pnl_cents(60, 10, won=False) == -10 * 60


def test_settle_match_updates_orders_and_bankroll(conn) -> None:
    db.record_bankroll(conn, 20000, "deposit")
    signal_id = db.log_signal(conn, _signal())
    db.log_order(conn, _order(signal_id))

    pnl = settlement.settle_match(conn, "999", won=True)
    assert pnl == 10 * (100 - 50)  # 500c

    row = conn.execute(
        "SELECT status, pnl_cents FROM orders WHERE id = 'ord1'"
    ).fetchone()
    assert row["status"] == "settled"
    assert row["pnl_cents"] == 500
    assert db.latest_bankroll(conn) == 20500


def test_settle_match_skips_unfilled(conn) -> None:
    db.record_bankroll(conn, 20000, "deposit")
    signal_id = db.log_signal(conn, _signal())
    db.log_order(conn, _order(signal_id, filled=None))  # never filled

    pnl = settlement.settle_match(conn, "999", won=True)
    assert pnl == 0
    assert db.latest_bankroll(conn) == 20000


def test_settle_match_loss(conn) -> None:
    db.record_bankroll(conn, 20000, "deposit")
    signal_id = db.log_signal(conn, _signal())
    db.log_order(conn, _order(signal_id))

    pnl = settlement.settle_match(conn, "999", won=False)
    assert pnl == -500
    assert db.latest_bankroll(conn) == 19500
