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


def _signal(match_id: str = "999:H:Brazil_Serbia") -> Signal:
    return Signal(
        match_id=match_id,
        market_ticker="KXWC26-BRA-H",
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


def test_ordered_tickers_includes_every_status(conn) -> None:
    # The no-re-bet guard must remember a market across settle/cancel, not just while the
    # Kalshi position is live. A cancelled (liquidated) order's ticker must still be returned.
    sid_filled = db.log_signal(conn, _signal("1:H:Brazil_Serbia"))
    db.log_order(conn, _order(sid_filled, order_id="ord_filled"))

    cancelled = _signal("2:A:France_Spain")
    cancelled["market_ticker"] = "KXWC26-ESP-A"
    sid_cancelled = db.log_signal(conn, cancelled)
    order = _order(sid_cancelled, order_id="ord_cancelled")
    order["status"] = "cancelled"
    db.log_order(conn, order)

    assert db.ordered_tickers(conn) == {"KXWC26-BRA-H", "KXWC26-ESP-A"}


def test_ordered_tickers_empty_when_no_orders(conn) -> None:
    # A signal with no order row must not appear — only markets we actually traded count.
    db.log_signal(conn, _signal("3:H:Brazil_Serbia"))
    assert db.ordered_tickers(conn) == set()


def test_settle_fixture_winning_outcome(conn) -> None:
    db.record_bankroll(conn, 20000, "deposit")
    signal_id = db.log_signal(conn, _signal("999:H:Brazil_Serbia"))
    db.log_order(conn, _order(signal_id))

    pnl = settlement.settle_fixture(conn, 999, "H")  # bet home, home won
    assert pnl == 10 * (100 - 50)  # 500c

    row = conn.execute(
        "SELECT status, pnl_cents FROM orders WHERE id = 'ord1'"
    ).fetchone()
    assert row["status"] == "settled"
    assert row["pnl_cents"] == 500
    # Settlement records per-order P&L only; bankroll is left to the Kalshi re-sync.
    assert db.latest_bankroll(conn) == 20000


def test_settle_fixture_losing_outcome(conn) -> None:
    db.record_bankroll(conn, 20000, "deposit")
    signal_id = db.log_signal(conn, _signal("999:A:Brazil_Serbia"))  # bet on away
    db.log_order(conn, _order(signal_id))

    pnl = settlement.settle_fixture(conn, 999, "H")  # home won -> away bet loses
    assert pnl == -500
    row = conn.execute("SELECT pnl_cents FROM orders WHERE id = 'ord1'").fetchone()
    assert row["pnl_cents"] == -500
    assert db.latest_bankroll(conn) == 20000  # settlement does not touch the ledger


def test_settle_fixture_draw_outcome_wins(conn) -> None:
    db.record_bankroll(conn, 20000, "deposit")
    signal_id = db.log_signal(conn, _signal("999:D:Brazil_Serbia"))  # bet on draw
    db.log_order(conn, _order(signal_id))

    pnl = settlement.settle_fixture(conn, 999, "D")  # draw -> draw bet wins
    assert pnl == 500
    assert db.latest_bankroll(conn) == 20000  # settlement does not touch the ledger


def test_real_peak_ignores_fallback_rows(conn) -> None:
    # Fallback syncs (placeholder bankroll when Kalshi balance can't be read) must not
    # count toward the stop-loss high-water mark.
    db.record_bankroll(conn, 20000, "sync_fallback")  # fake default — must be ignored
    db.record_bankroll(conn, 5410, "sync")  # real high-water
    db.record_bankroll(conn, 4691, "sync")  # real, after a loss
    assert db.real_peak_bankroll(conn) == 5410


def test_real_peak_none_without_real_sync(conn) -> None:
    db.record_bankroll(conn, 10000, "sync_fallback")
    assert db.real_peak_bankroll(conn) is None


def test_settle_fixture_skips_unfilled(conn) -> None:
    db.record_bankroll(conn, 20000, "deposit")
    signal_id = db.log_signal(conn, _signal())
    db.log_order(conn, _order(signal_id, filled=None))  # never filled

    pnl = settlement.settle_fixture(conn, 999, "H")
    assert pnl == 0
    assert db.latest_bankroll(conn) == 20000
