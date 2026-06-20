"""SQLite database layer — schema creation and connection helper (PRD section 10).

Datetimes are stored as ISO-8601 TEXT (e.g. "2026-06-19T18:00:00Z") for lexical
ordering. Run `python -m data.db` to (re)create the schema.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from config import settings

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
    match_id        TEXT NOT NULL REFERENCES matches(id),
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


if __name__ == "__main__":
    from config import configure_logging

    configure_logging()
    init_db()
