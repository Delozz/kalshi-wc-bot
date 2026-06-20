"""Kelly sizing (strategy/kelly.py).

L6 — Half-Kelly Always: live sizing uses ``kelly_fraction * 0.5`` AND a hard cap of
``MAX_BET_FRACTION * bankroll``. Full Kelly is never used. Sizing must only ever run on
CALIBRATED probabilities (L5) — uncalibrated, overconfident inputs cause catastrophic
overbetting.

For a Kalshi YES contract bought at ``price`` (dollars, 0..1) that settles at $1.00 on
a win: net odds ``b = (1 - price) / price``, and full-Kelly fraction
``f* = p - (1 - p) / b`` where ``p`` is the (calibrated) model probability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sizing:
    """Result of a sizing decision (fractions of bankroll, plus dollar bet size)."""

    full_kelly: float
    used_fraction: float
    bet_size: float


def kelly_fraction(model_prob: float, price: float) -> float:
    """Full-Kelly fraction for a YES contract at ``price``; 0.0 if there is no edge."""
    if not (0.0 < price < 1.0) or not (0.0 <= model_prob <= 1.0):
        return 0.0
    b = (1.0 - price) / price
    edge_term = (b * model_prob - (1.0 - model_prob)) / b
    return max(edge_term, 0.0)


def half_kelly_size(
    model_prob: float,
    price: float,
    bankroll: float,
    *,
    fraction: float | None = None,
    max_bet_fraction: float | None = None,
) -> Sizing:
    """Half-Kelly bet size with the hard per-bet cap enforced (L6)."""
    frac_mult = settings.kelly_fraction if fraction is None else fraction
    cap = settings.max_bet_fraction if max_bet_fraction is None else max_bet_fraction

    full = kelly_fraction(model_prob, price)
    used = max(min(full * frac_mult, cap), 0.0)
    return Sizing(
        full_kelly=full, used_fraction=used, bet_size=used * max(bankroll, 0.0)
    )
