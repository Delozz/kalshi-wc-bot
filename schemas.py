"""Shared typed records (L7). TypedDicts mirror the SQLite schema in PRD section 10.

Named `schemas` rather than `types` to avoid shadowing the Python stdlib `types`
module, since the project root sits on sys.path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

Outcome = Literal["H", "D", "A"]
Stage = Literal["group", "round_of_16", "qf", "sf", "final"]
OrderSide = Literal["YES", "NO"]
OrderStatus = Literal["pending", "filled", "canceled"]
BankrollEvent = Literal["deposit", "win", "loss", "fee"]


class MatchRecord(TypedDict):
    """A single match — mirrors the `matches` table."""

    id: str  # "{home}_{away}_{date}"
    home_team: str
    away_team: str
    kickoff_utc: datetime
    stage: Stage
    result: Outcome | None  # null until settled
    home_goals: int | None
    away_goals: int | None


class FeatureVector(TypedDict):
    """Engineered features for one match — mirrors the `features` table."""

    match_id: str
    computed_at: datetime  # must be < kickoff_utc (enforced by lookahead_guard)
    elo_delta: float
    form_5_home: float
    form_5_away: float
    pinnacle_implied_home: float
    pinnacle_implied_draw: float
    pinnacle_implied_away: float
    xg_delta_home: float
    xg_delta_away: float
    fifa_rank_delta: int


class Signal(TypedDict):
    """A betting signal — mirrors the `signals` table."""

    match_id: str
    market_ticker: str
    side: OrderSide
    model_prob: float
    market_implied: float  # kalshi YES price
    edge: float  # model_prob - market_implied
    kelly_fraction: float
    bet_size_cents: int
    generated_at: datetime


class Order(TypedDict):
    """An order placed on Kalshi — mirrors the `orders` table."""

    id: str  # Kalshi order_id
    signal_id: int
    status: OrderStatus
    limit_price: float
    contracts: int
    filled_price: float | None
    placed_at: datetime
    settled_at: datetime | None
    pnl_cents: int | None


class BankrollEntry(TypedDict):
    """A bankroll ledger entry — mirrors the `bankroll_log` table."""

    timestamp: datetime
    balance_cents: int
    event: BankrollEvent
