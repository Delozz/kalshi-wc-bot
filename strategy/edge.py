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


def apply_lineup_prior(
    probs: dict[str, float],
    lineup_delta: float,
    *,
    weight: float | None = None,
) -> dict[str, float]:
    """Tilt the full {H, D, A} vector toward the stronger announced XI, renormalized.

    ``lineup_delta`` is the home-minus-away starting-XI strength in ``[-1, 1]`` (0.0 until
    lineups are announced — identity, the zero-impact fallback). Same mechanic as
    :func:`apply_squad_prior`, weighted by ``settings.lineup_weight``. This replaced the
    old per-leg nudge, which scaled a single leg in isolation and could leave the three
    legs summing above 1 — edge fabricated by the adjustment itself.
    """
    w = settings.lineup_weight if weight is None else weight
    return apply_squad_prior(probs, lineup_delta, weight=w)


def apply_squad_prior(
    probs: dict[str, float],
    squad_delta: float,
    *,
    weight: float | None = None,
) -> dict[str, float]:
    """Tilt the full {H, D, A} probability vector toward the stronger squad, renormalized.

    ``squad_delta`` is the home-minus-away squad strength in ``[-1, 1]`` (positive favours
    home). The favoured side's win probability is scaled up by ``weight * |delta|`` while
    *both* the draw and the underdog are scaled down by the same factor, then the vector is
    renormalized to sum to 1. Scaling-then-renormalizing keeps the result a valid, still
    reasonably-calibrated distribution (no probability can go negative or exceed 1), unlike
    the per-leg lineup nudge which leaves the draw untouched.

    With ``squad_delta == 0`` (either squad unrated) this is the identity — the zero-impact
    fallback. ``weight`` defaults to ``settings.squad_weight``.
    """
    w = settings.squad_weight if weight is None else weight
    tilt = w * squad_delta
    if tilt == 0.0 or not probs:
        return dict(probs)

    favorite = "H" if tilt > 0 else "A"
    # Cap the shrink factor just below 1 so the de-emphasized legs stay strictly positive.
    magnitude = min(abs(tilt), 0.95)

    scaled = {
        outcome: prob * (1.0 + magnitude if outcome == favorite else 1.0 - magnitude)
        for outcome, prob in probs.items()
    }
    total = sum(scaled.values())
    if total <= 0.0:
        return dict(probs)
    return {outcome: value / total for outcome, value in scaled.items()}


def apply_confederation_prior(
    probs: dict[str, float],
    elo_delta: float,
    *,
    weight: float | None = None,
) -> dict[str, float]:
    """Tilt the H/D/A vector for inter-confederation ELO drift, then renormalize.

    ``elo_delta`` is the home-minus-away confederation ELO offset (signed points, from
    ``features.confederation.elo_delta``). It is applied the way ELO itself treats a rating
    gap: the home/away win-odds ratio is multiplied by ``10 ** (weight * elo_delta / 400)``,
    realized by scaling the home leg by ``10 ** (w*d/800)`` and the away leg by the inverse
    while the draw is held, then renormalizing. So ``P_H/P_A`` shifts by exactly the ELO
    factor and the result stays a valid distribution. This corrects the model *output*
    (engine-agnostic) rather than the rating prior, which the Dixon-Coles MLE was shown to
    override for data-rich teams.

    With ``elo_delta == 0`` (same confederation, or either team unmapped) or ``weight == 0``
    this is the identity — the zero-impact fallback. ``weight`` defaults to
    ``settings.confederation_weight``; only the H and A legs are scaled, any other key is
    carried through unchanged.
    """
    w = settings.confederation_weight if weight is None else weight
    if w == 0.0 or elo_delta == 0.0 or not probs:
        return dict(probs)

    root = 10.0 ** (w * elo_delta / 800.0)  # sqrt of the 10**(d/400) odds factor
    scaled = {
        outcome: (
            prob * root if outcome == "H" else prob / root if outcome == "A" else prob
        )
        for outcome, prob in probs.items()
    }
    total = sum(scaled.values())
    if total <= 0.0:
        return dict(probs)
    return {outcome: value / total for outcome, value in scaled.items()}


def blend_with_book(
    probs: dict[str, float],
    anchor: dict[str, float] | None,
    *,
    weight: float | None = None,
) -> dict[str, float]:
    """Shrink the model's H/D/A vector toward a market anchor, renormalized.

    ``p_final = w * p_model + (1 - w) * p_anchor`` per outcome, where ``w`` is the
    *model's* share (``settings.model_blend_weight``, default 0.30 — the anchor carries
    70%). The anchor is the sportsbook no-vig consensus when available, else normalized
    Kalshi prices. This is the core anti-phantom-edge control: 18 settled live bets showed
    the line's Brier (0.077) beating the raw model's (0.144), so an "edge" that exists only
    in the raw model is far likelier mis-calibration than value. After blending, an edge
    survives only where the anchor itself diverges from the Kalshi price (book-vs-Kalshi
    mispricing) and/or the model's residual disagreement is large.

    Identity (returns ``probs`` unchanged) when the anchor is missing, doesn't cover every
    outcome the model prices, or ``weight >= 1.0`` — the zero-impact fallback.
    """
    w = settings.model_blend_weight if weight is None else weight
    if not probs or not anchor or w >= 1.0:
        return dict(probs)
    if any(outcome not in anchor for outcome in probs):
        return dict(probs)
    blended = {
        outcome: w * prob + (1.0 - w) * float(anchor[outcome])
        for outcome, prob in probs.items()
    }
    total = sum(blended.values())
    if total <= 0.0:
        return dict(probs)
    return {outcome: value / total for outcome, value in blended.items()}


def build_signal(
    *,
    match_id: str,
    market_ticker: str,
    model_prob: float,
    kalshi_yes_price: float,
    bankroll: float,
    side: OrderSide = "YES",
    threshold: float | None = None,
) -> Signal | None:
    """Build a sized :class:`Signal` if the edge clears the threshold, else ``None``.

    ``model_prob`` arrives fully adjusted (confederation/squad/lineup tilts and the
    market-anchor blend are applied upstream in ``signal_gen``); this only checks the
    edge and sizes the bet.
    """
    edge = compute_edge(model_prob, kalshi_yes_price)
    if not has_edge(edge, threshold=threshold):
        logger.debug("No signal for %s: edge %.3f below threshold", market_ticker, edge)
        return None

    sizing = kelly.half_kelly_size(model_prob, kalshi_yes_price, bankroll)
    return Signal(
        match_id=match_id,
        market_ticker=market_ticker,
        side=side,
        model_prob=model_prob,
        market_implied=kalshi_yes_price,
        edge=edge,
        kelly_fraction=sizing.used_fraction,
        bet_size_cents=round(sizing.bet_size * 100),
        generated_at=datetime.now(timezone.utc),
    )
