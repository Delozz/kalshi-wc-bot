"""Portfolio tracker (execution/portfolio.py).

Holds bankroll, peak bankroll (for the stop-loss), and open positions, and exposes the
exposure figures the risk checks need. Syncs from Kalshi's balance/positions endpoints
at startup; the peak is tracked so a drawdown can be measured against it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ingestion import kalshi

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Position:
    ticker: str
    side: str
    count: int
    avg_price_cents: int

    @property
    def exposure_cents(self) -> int:
        return self.count * self.avg_price_cents


@dataclass
class PortfolioState:
    bankroll_cents: int
    peak_bankroll_cents: int
    positions: list[Position] = field(default_factory=list)
    # True when the Kalshi balance could not be read and a fallback figure is in use; such
    # a balance must never be recorded as a real high-water mark for the stop-loss.
    balance_is_fallback: bool = False

    @property
    def exposure_cents(self) -> int:
        return sum(p.exposure_cents for p in self.positions)

    @property
    def open_count(self) -> int:
        return len(self.positions)

    def update_bankroll(self, balance_cents: int) -> None:
        """Set the current bankroll and ratchet the peak upward."""
        self.bankroll_cents = balance_cents
        self.peak_bankroll_cents = max(self.peak_bankroll_cents, balance_cents)


def total_exposure_cents(positions: list[Position]) -> int:
    """Sum of cost basis across open positions (dollars at risk, in cents)."""
    return sum(p.exposure_cents for p in positions)


def ratchet_peak(
    state: PortfolioState, historical_peak_cents: int | None
) -> PortfolioState:
    """Raise ``state.peak_bankroll_cents`` to the true high-water mark.

    ``sync_from_kalshi`` initialises peak to the *current* balance, which alone can never
    register a drawdown across runs (peak always equals bankroll). Folding in the real
    high-water mark from the ledger restores the stop-loss. A fallback balance never lifts
    the peak — only the historical real peak applies in that case (a fake figure must not
    raise the bar the stop-loss measures against)."""
    hist = historical_peak_cents or 0
    if state.balance_is_fallback:
        # The current balance is a placeholder; discard it and keep only the real peak.
        state.peak_bankroll_cents = hist
    else:
        state.peak_bankroll_cents = max(state.bankroll_cents, hist)
    return state


def _parse_positions(raw: dict[str, object] | None) -> list[Position]:
    if not raw or not isinstance(raw, dict):
        return []
    market_positions = raw.get("market_positions", [])
    positions: list[Position] = []
    if not isinstance(market_positions, list):
        return positions
    for item in market_positions:
        if not isinstance(item, dict):
            continue
        # position_fp (FixedPointCount string, +YES/-NO); legacy: position (int)
        pos_raw = item.get("position_fp") or item.get("position")
        count = round(float(pos_raw)) if pos_raw else 0
        if count == 0:
            continue
        # market_exposure_dollars (dollar string → cents); legacy: market_exposure (cents int)
        exp_raw = item.get("market_exposure_dollars")
        exposure_cents = (
            round(float(exp_raw) * 100)
            if exp_raw is not None
            else int(item.get("market_exposure", 0))
        )
        positions.append(
            Position(
                ticker=str(item.get("ticker", "")),
                side="yes" if count > 0 else "no",
                count=abs(count),
                avg_price_cents=exposure_cents // max(abs(count), 1),
            )
        )
    return positions


async def sync_from_kalshi(*, fallback_bankroll_cents: int) -> PortfolioState:
    """Build portfolio state from Kalshi, falling back to a starting bankroll (L9)."""
    balance = await kalshi.get_balance()
    positions_raw = await kalshi.get_positions()

    if balance and isinstance(balance, dict) and "balance" in balance:
        bankroll = int(balance["balance"])
        is_fallback = False
    else:
        logger.warning("Could not read Kalshi balance; using fallback bankroll")
        bankroll = fallback_bankroll_cents
        is_fallback = True

    state = PortfolioState(
        bankroll_cents=bankroll,
        peak_bankroll_cents=bankroll,
        positions=_parse_positions(positions_raw),
        balance_is_fallback=is_fallback,
    )
    logger.info(
        "Portfolio synced: bankroll=%dc, %d open positions, exposure=%dc",
        state.bankroll_cents,
        state.open_count,
        state.exposure_cents,
    )
    return state
