"""Backtest performance metrics (backtest/metrics.py).

Sharpe here is a per-bet ratio (mean / std of per-bet returns), not annualized — the
WC bet stream is event-driven, not daily, so a per-bet figure is the honest summary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestMetrics:
    n_bets: int
    hit_rate: float
    total_pnl: float
    roi: float
    sharpe: float
    max_drawdown: float
    final_bankroll: float


def per_bet_sharpe(returns: Sequence[float]) -> float:
    """Mean / standard deviation of per-bet returns (0.0 if undefined)."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    return mean / std if std > 0 else 0.0


def max_drawdown(equity: Sequence[float]) -> float:
    """Largest peak-to-trough fractional drop in the equity curve."""
    peak = -math.inf
    mdd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            mdd = max(mdd, (peak - value) / peak)
    return mdd


def compute_metrics(
    returns: Sequence[float],
    pnls: Sequence[float],
    wins: Sequence[bool],
    equity_curve: Sequence[float],
    initial_bankroll: float,
) -> BacktestMetrics:
    """Aggregate per-bet results into summary metrics."""
    n = len(pnls)
    total_pnl = sum(pnls)
    return BacktestMetrics(
        n_bets=n,
        hit_rate=(sum(1 for w in wins if w) / n) if n else 0.0,
        total_pnl=total_pnl,
        roi=(total_pnl / initial_bankroll) if initial_bankroll else 0.0,
        sharpe=per_bet_sharpe(returns),
        max_drawdown=max_drawdown(equity_curve),
        final_bankroll=initial_bankroll + total_pnl,
    )
