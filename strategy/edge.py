"""Edge detection (strategy/edge.py).

edge = model_prob - kalshi_yes_price. A bet is only considered when the edge clears the
minimum threshold (default 4%, never lowered without backtesting justification). Signal
assembly applies half-Kelly sizing (L6) on top of the edge check.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from config import settings
from schemas import OrderSide, Signal
from strategy import kelly

logger = logging.getLogger(__name__)


def compute_edge(model_prob: float, kalshi_yes_price: float) -> float:
    """Edge as a decimal: model probability minus the Kalshi YES price (0..1)."""
    return model_prob - kalshi_yes_price


def has_edge(edge: float, *, threshold: float | None = None) -> bool:
    """True if the edge meets the minimum threshold."""
    thr = settings.min_edge_threshold if threshold is None else threshold
    return edge >= thr


def apply_lineup_adjustment(model_prob: float, lineup_delta: float) -> float:
    """Nudge a model probability by an already-signed lineup strength delta, clamped 0–1.

    ``lineup_delta`` is the home-minus-away starting-XI strength for *this* outcome
    (signed upstream: positive favours the bet, negative opposes it). It is 0.0 whenever
    lineups are not yet announced, so this is the identity then (zero-impact fallback).
    """
    adjusted = model_prob * (1.0 + settings.lineup_weight * lineup_delta)
    return max(0.0, min(1.0, adjusted))


def build_signal(
    *,
    match_id: str,
    market_ticker: str,
    model_prob: float,
    kalshi_yes_price: float,
    bankroll: float,
    side: OrderSide = "YES",
    threshold: float | None = None,
    lineup_delta: float = 0.0,
) -> Signal | None:
    """Build a sized :class:`Signal` if the edge clears the threshold, else ``None``.

    When ``lineup_delta`` is non-zero (lineups announced), the model probability is
    nudged before the edge check; with the default 0.0 the behaviour is unchanged.
    """
    adjusted_prob = apply_lineup_adjustment(model_prob, lineup_delta)
    edge = compute_edge(adjusted_prob, kalshi_yes_price)
    if not has_edge(edge, threshold=threshold):
        logger.debug("No signal for %s: edge %.3f below threshold", market_ticker, edge)
        return None

    sizing = kelly.half_kelly_size(adjusted_prob, kalshi_yes_price, bankroll)
    return Signal(
        match_id=match_id,
        market_ticker=market_ticker,
        side=side,
        model_prob=adjusted_prob,
        market_implied=kalshi_yes_price,
        edge=edge,
        kelly_fraction=sizing.used_fraction,
        bet_size_cents=round(sizing.bet_size * 100),
        generated_at=datetime.now(timezone.utc),
    )
