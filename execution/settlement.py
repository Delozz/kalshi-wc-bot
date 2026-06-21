"""Settlement (execution/settlement.py).

Closes the trade lifecycle: given a match outcome, settle each YES order on it, mark the
orders settled with realized P&L, and post the new balance to the bankroll ledger. A YES
contract bought at ``filled_price`` cents pays 100c on a win and 0c on a loss.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

from data.db import (
    latest_bankroll,
    record_bankroll,
    settle_order,
    unsettled_orders_for_fixture,
)

logger = logging.getLogger(__name__)


def compute_pnl_cents(filled_price_cents: int, contracts: int, won: bool) -> int:
    """Realized P&L in cents for a YES position (pays 100c on win, 0c on loss)."""
    if won:
        return contracts * (100 - filled_price_cents)
    return contracts * (-filled_price_cents)


def settle_position(
    conn: sqlite3.Connection,
    *,
    order_id: str,
    filled_price_cents: int,
    contracts: int,
    won: bool,
) -> int:
    """Settle one order: write P&L, mark it settled, and update the bankroll ledger."""
    pnl = compute_pnl_cents(filled_price_cents, contracts, won)
    settle_order(conn, order_id, settled_at=datetime.now(timezone.utc), pnl_cents=pnl)
    base = latest_bankroll(conn)
    if base is not None:
        record_bankroll(conn, base + pnl, "win" if pnl >= 0 else "loss")
    else:
        logger.warning("No bankroll baseline to update for settled order %s", order_id)
    logger.info("Settled %s: pnl=%dc (won=%s)", order_id, pnl, won)
    return pnl


def _outcome_of(match_id: str) -> str | None:
    """Read the bet outcome (H/D/A) from a ``"{fixture_id}:{outcome}:..."`` match_id."""
    parts = match_id.split(":")
    return parts[1] if len(parts) >= 2 else None


def settle_fixture(conn: sqlite3.Connection, fixture_id: object, result: str) -> int:
    """Settle every unsettled order for a fixture against the match ``result`` (H/D/A).

    Each order wins iff the outcome it bet on equals ``result``. Returns total realized
    P&L in cents.
    """
    total = 0
    for row in unsettled_orders_for_fixture(conn, str(fixture_id)):
        filled = row["filled_price"]
        if filled is None:
            logger.info("Order %s never filled; skipping settlement", row["id"])
            continue
        outcome = _outcome_of(row["match_id"])
        if outcome is None:
            logger.warning("Cannot read outcome from match_id %s", row["match_id"])
            continue
        total += settle_position(
            conn,
            order_id=row["id"],
            filled_price_cents=round(float(filled) * 100),
            contracts=int(row["contracts"]),
            won=outcome == result,
        )
    return total
