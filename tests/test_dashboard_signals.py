"""Tests for the dashboard's recent-signals panel (dashboard/app._recent_signals).

Pins the fix that scopes the panel to the latest generation cycle, so stale signals
from earlier runs (possibly sized at a different bankroll) never leak into the live
monitor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import data.db as db
from dashboard import app


def _insert(conn, ticker: str, bet_cents: int, generated_at: datetime) -> None:
    conn.execute(
        "INSERT INTO signals (match_id, market_ticker, side, model_prob, "
        "market_implied, edge, kelly_fraction, bet_size_cents, generated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("m", ticker, "YES", 0.2, 0.1, 0.1, 0.5, bet_cents, generated_at.isoformat()),
    )


def _seed(path) -> None:
    db.init_db(path)
    conn = db.connect(path)
    now = datetime.now(timezone.utc)
    _insert(conn, "STALE", 500, now - timedelta(hours=4))  # earlier run, big bet
    _insert(conn, "FRESH-A", 192, now)
    _insert(conn, "FRESH-B", 269, now - timedelta(seconds=1))
    conn.commit()
    conn.close()


def test_recent_signals_only_latest_cycle(tmp_path, monkeypatch) -> None:
    path = tmp_path / "t.sqlite"
    _seed(path)
    original = db.connect
    monkeypatch.setattr(db, "connect", lambda db_path=None: original(path))

    tickers = {s["market_ticker"] for s in app._recent_signals()}
    assert tickers == {"FRESH-A", "FRESH-B"}
    assert "STALE" not in tickers  # the 4-hours-old big-bet row is excluded


def test_recent_signals_empty_db_returns_empty(tmp_path, monkeypatch) -> None:
    path = tmp_path / "empty.sqlite"
    db.init_db(path)
    original = db.connect
    monkeypatch.setattr(db, "connect", lambda db_path=None: original(path))

    assert app._recent_signals() == []
