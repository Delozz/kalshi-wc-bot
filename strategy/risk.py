"""Risk controls (strategy/risk.py).

Hard guards that sit between a signal and an order. None of these may be removed during
live trading (the stop-loss especially). Limits default to the project standards:
stop-loss at 25% drawdown from peak, 20% max portfolio exposure, at most 3 open
positions, and a minimum market liquidity floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)

MAX_OPEN_POSITIONS: int = 3
MIN_OPEN_INTEREST: float = 5000.0  # dollars
MAX_PRICE_MOVE: float = 0.03  # cancel a signal if the price drifts more than 3 cents


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
