"""SQLite database layer — schema creation and connection helper (PRD section 10).

Datetimes are stored as ISO-8601 TEXT (e.g. "2026-06-19T18:00:00Z") for lexical
ordering. Run `python -m data.db` to (re)create the schema.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from schemas import Order, Signal

logger = logging.getLogger(__name__)

SCHEMA: str = """
CREATE TABLE IF NOT EXISTS matches (
    id          TEXT PRIMARY KEY,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    kickoff_utc TEXT NOT NULL,
    stage       TEXT,
    result      TEXT,
    home_goals  INTEGER,
    away_goals  INTEGER
);

CREATE TABLE IF NOT EXISTS features (
    match_id              TEXT NOT NULL REFERENCES matches(id),
    computed_at           TEXT NOT NULL,
    elo_delta             REAL,
    form_5_home           REAL,
    form_5_away           REAL,
    pinnacle_implied_home REAL,
    pinnacle_implied_draw REAL,
    pinnacle_implied_away REAL,
    xg_delta_home         REAL,
    xg_delta_away         REAL,
    fifa_rank_delta       INTEGER,
    PRIMARY KEY (match_id, computed_at)
);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        TEXT NOT NULL,
    market_ticker   TEXT,
    side            TEXT,
    model_prob      REAL,
    market_implied  REAL,
    edge            REAL,
    kelly_fraction  REAL,
    bet_size_cents  INTEGER,
    generated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id           TEXT PRIMARY KEY,
    signal_id    INTEGER REFERENCES signals(id),
    status       TEXT,
    limit_price  REAL,
    contracts    INTEGER,
    filled_price REAL,
    placed_at    TEXT,
    settled_at   TEXT,
    pnl_cents    INTEGER
);

CREATE TABLE IF NOT EXISTS bankroll_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    balance_cents INTEGER NOT NULL,
    event         TEXT
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys on and Row factory set."""
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create all tables if they do not already exist."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", db_path or settings.db_path)


def _iso(value: object) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def log_signal(conn: sqlite3.Connection, signal: Signal) -> int:
    """Insert a generated signal into the `signals` table; return its row id."""
    cursor = conn.execute(
        "INSERT INTO signals (match_id, market_ticker, side, model_prob, market_implied, "
        "edge, kelly_fraction, bet_size_cents, generated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            signal["match_id"],
            signal["market_ticker"],
            signal["side"],
            signal["model_prob"],
            signal["market_implied"],
            signal["edge"],
            signal["kelly_fraction"],
            signal["bet_size_cents"],
            _iso(signal["generated_at"]),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def log_order(conn: sqlite3.Connection, order: Order) -> None:
    """Insert or replace an order row."""
    conn.execute(
        "INSERT OR REPLACE INTO orders (id, signal_id, status, limit_price, contracts, "
        "filled_price, placed_at, settled_at, pnl_cents) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            order["id"],
            order["signal_id"],
            order["status"],
            order["limit_price"],
            order["contracts"],
            order.get("filled_price"),
            _iso(order["placed_at"]),
            _iso(order.get("settled_at")),
            order.get("pnl_cents"),
        ),
    )
    conn.commit()


def update_order_status(
    conn: sqlite3.Connection,
    order_id: str,
    status: str,
    *,
    filled_price: float | None = None,
) -> None:
    """Update an order's status (and filled price if provided)."""
    conn.execute(
        "UPDATE orders SET status = ?, filled_price = COALESCE(?, filled_price) "
        "WHERE id = ?",
        (status, filled_price, order_id),
    )
    conn.commit()


def settle_order(
    conn: sqlite3.Connection, order_id: str, *, settled_at: object, pnl_cents: int
) -> None:
    """Mark an order settled with its realized P&L."""
    conn.execute(
        "UPDATE orders SET status = 'settled', settled_at = ?, pnl_cents = ? WHERE id = ?",
        (_iso(settled_at), pnl_cents, order_id),
    )
    conn.commit()


def record_bankroll(conn: sqlite3.Connection, balance_cents: int, event: str) -> None:
    """Append a bankroll ledger entry."""
    conn.execute(
        "INSERT INTO bankroll_log (timestamp, balance_cents, event) VALUES (?,?,?)",
        (datetime.now(timezone.utc).isoformat(), balance_cents, event),
    )
    conn.commit()


def latest_bankroll(conn: sqlite3.Connection) -> int | None:
    """Most recent recorded bankroll balance in cents, or None if the ledger is empty."""
    row = conn.execute(
        "SELECT balance_cents FROM bankroll_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return int(row["balance_cents"]) if row else None


def unsettled_orders_for_match(
    conn: sqlite3.Connection, match_id_suffix: str
) -> list[sqlite3.Row]:
    """Open (unsettled) orders whose signal's match_id ends with the suffix."""
    return conn.execute(
        "SELECT o.* FROM orders o JOIN signals s ON o.signal_id = s.id "
        "WHERE s.match_id LIKE ? AND o.status != 'settled'",
        (f"%{match_id_suffix}",),
    ).fetchall()


if __name__ == "__main__":
    from config import configure_logging

    configure_logging()
    init_db()
