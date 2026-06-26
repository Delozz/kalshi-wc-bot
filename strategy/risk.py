"""Risk controls (strategy/risk.py).

Hard guards that sit between a signal and an order. None of these may be removed during
live trading (the stop-loss especially). Limits default to the project standards:
stop-loss at 25% drawdown from peak, 20% max portfolio exposure, at most 5 open
positions, and a minimum market liquidity floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)

MAX_OPEN_POSITIONS: int = 5
MIN_OPEN_INTEREST: float = 5000.0  # dollars
MAX_PRICE_MOVE: float = 0.03  # cancel a signal if the price drifts more than 3 cents

# Signal-quality guards. These reject draw/underdog legs that are really model
# mis-calibration against a strong favorite (the France-vs-Iraq "bet the upset/draw"
# problem), where the sharp Kalshi line is better-informed than our ELO model. They run
# in signal_gen's candidate phase, before edge/sizing, so a doomed leg never claims a
# ranking slot.
MIN_MARKET_PRICE: float = 0.06  # below 6c the edge is longshot noise, not signal
MAX_MODEL_MARKET_RATIO: float = 2.5  # model_prob may not exceed 2.5x the market price
POWERHOUSE_ELO_GAP: float = 200.0  # ELO gap above which draw/upset legs are untrusted
# Lower bar applied only when the squad-strength prior agrees on the same favorite: a
# moderate ELO edge plus a stronger squad is enough to distrust draw/upset legs (catches
# the Portugal-type favorite whose ELO gap sits just under the pure-ELO threshold).
POWERHOUSE_ELO_GAP_WITH_SQUAD: float = 150.0


def stop_loss_triggered(
    bankroll: float, peak_bankroll: float, *, threshold: float | None = None
) -> bool:
    """True if bankroll has fallen at least ``threshold`` below its peak."""
    thr = settings.stop_loss_threshold if threshold is None else threshold
    if peak_bankroll <= 0:
        return False
    return bankroll <= peak_bankroll * (1.0 - thr)


def exposure_ok(
    open_exposure: float,
    new_bet: float,
    bankroll: float,
    *,
    max_exposure: float | None = None,
) -> bool:
    """True if adding ``new_bet`` keeps total exposure within the portfolio cap."""
    cap = settings.max_portfolio_exposure if max_exposure is None else max_exposure
    if bankroll <= 0:
        return False
    return (open_exposure + new_bet) <= cap * bankroll


def position_count_ok(n_open: int, *, max_positions: int = MAX_OPEN_POSITIONS) -> bool:
    """True if there is room for another open position."""
    return n_open < max_positions


def liquidity_ok(open_interest: float, *, min_oi: float = MIN_OPEN_INTEREST) -> bool:
    """True if a market has enough open interest to trade into."""
    return open_interest >= min_oi


def price_stable(
    signal_price: float, current_price: float, *, max_move: float = MAX_PRICE_MOVE
) -> bool:
    """True if the market has not drifted more than ``max_move`` since the signal."""
    return abs(current_price - signal_price) <= max_move


def price_floor_ok(market_price: float, *, min_price: float = MIN_MARKET_PRICE) -> bool:
    """True if the market price clears the longshot floor (below it, edge is noise)."""
    return market_price >= min_price


def mismatch_ok(
    model_prob: float,
    market_price: float,
    *,
    max_ratio: float = MAX_MODEL_MARKET_RATIO,
) -> bool:
    """True unless the model probability dwarfs the market price (calibration outlier).

    A model that prices an outcome at more than ``max_ratio`` times the sharp Kalshi line
    is far likelier mis-calibrated than to have found genuine edge — so we defer to the
    market. ``market_price <= 0`` is never tradeable.
    """
    if market_price <= 0.0:
        return False
    return (model_prob / market_price) <= max_ratio


def favorite_not_overwhelming(
    bet_on_favorite: bool,
    favorite_elo_gap: float,
    *,
    max_gap: float = POWERHOUSE_ELO_GAP,
) -> bool:
    """True unless this is a draw/underdog leg against an overwhelming ELO favorite.

    When one side outrates the other by ``max_gap`` ELO or more, the model's draw and
    upset probabilities are untrustworthy (it under-rates true mismatches the market
    prices sharply). The favorite's own win leg is always allowed through.
    """
    if bet_on_favorite:
        return True
    return favorite_elo_gap < max_gap


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str


def check_all(
    *,
    bankroll: float,
    peak_bankroll: float,
    open_exposure: float,
    new_bet: float,
    n_open: int,
    open_interest: float,
) -> RiskDecision:
    """Run every guard in priority order; return the first failure or approval."""
    if stop_loss_triggered(bankroll, peak_bankroll):
        return RiskDecision(False, "stop_loss")
    if not position_count_ok(n_open):
        return RiskDecision(False, "max_positions")
    if not liquidity_ok(open_interest):
        return RiskDecision(False, "insufficient_liquidity")
    if not exposure_ok(open_exposure, new_bet, bankroll):
        return RiskDecision(False, "exposure_cap")
    return RiskDecision(True, "ok")


def outcome_admissible(
    *,
    bet_on_favorite: bool,
    model_prob: float,
    market_price: float,
    favorite_elo_gap: float,
    squad_confirms_favorite: bool = False,
) -> RiskDecision:
    """Pre-edge signal-quality gate run per outcome before edge/sizing.

    Rejects legs the market is better-informed on: longshots below the price floor,
    calibration outliers where the model dwarfs the line, and draw/upset legs against a
    clear favorite. The favorite check has two tiers: an overwhelming pure-ELO gap, or a
    merely-strong ELO gap that the squad-strength prior independently confirms
    (``squad_confirms_favorite``) — the latter catches Portugal-type favorites whose ELO
    edge alone sits just under the pure threshold. Applied in ``signal_gen`` so a doomed
    leg never claims a ranking slot, and so we stop betting upsets like Iraq-over-France.
    """
    if not price_floor_ok(market_price):
        return RiskDecision(False, "below_price_floor")
    if not mismatch_ok(model_prob, market_price):
        return RiskDecision(False, "model_market_mismatch")
    if not favorite_not_overwhelming(bet_on_favorite, favorite_elo_gap):
        return RiskDecision(False, "powerhouse_favorite")
    if (
        not bet_on_favorite
        and squad_confirms_favorite
        and favorite_elo_gap >= POWERHOUSE_ELO_GAP_WITH_SQUAD
    ):
        return RiskDecision(False, "powerhouse_favorite_squad")
    return RiskDecision(True, "ok")
