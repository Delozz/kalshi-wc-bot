"""Simulated fills at historical prices (backtest/simulator.py).

Models a Kalshi-style YES contract: buy at an effective ask price (fair price plus
half the assumed spread, PRD section 7.3); it settles at $1.00 if the event occurs,
else $0.00. Prices and PnL are in dollars per contract.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SPREAD: float = 0.02  # 2-cent conservative spread (PRD 7.3)


@dataclass(frozen=True)
class Fill:
    """A simulated contract fill and its settled PnL."""

    price: float  # effective entry price (dollars per contract)
    contracts: int
    won: bool
    pnl: float  # dollars


def simulate_yes_fill(
    fair_price: float,
    won: bool,
    *,
    contracts: int = 1,
    spread: float = DEFAULT_SPREAD,
) -> Fill:
    """Simulate buying ``contracts`` YES at ``fair_price`` and settling the outcome."""
    effective = min(max(fair_price + spread / 2.0, 0.0), 1.0)
    pnl = contracts * ((1.0 - effective) if won else -effective)
    return Fill(price=effective, contracts=contracts, won=won, pnl=pnl)
