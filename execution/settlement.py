"""Settlement (execution/settlement.py).

Closes the trade lifecycle: given a match outcome, settle each YES order on it and mark
the orders settled with realized P&L. A YES contract bought at ``filled_price`` cents pays
100c on a win and 0c on a loss.

The bankroll ledger is intentionally NOT updated here. Live bankroll is authoritative from
Kalshi's balance endpoint (purchase cash leaves the account at fill time), so the settle
job re-syncs the real balance after settling — see ``scheduler.jobs._settle_finished``.
Adjusting the ledger additively here would double-count the cost basis that the synced
balance already reflects. ``pnl_cents`` on each order is the per-trade record for metrics.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

from data.db import settle_order, unsettled_orders_for_fixture

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
    """Settle one order: record realized P&L and mark it settled. Returns the P&L cents.

    Does not touch the bankroll ledger (Kalshi balance is authoritative; the settle job
    re-syncs it afterward).
    """
    pnl = compute_pnl_cents(filled_price_cents, contracts, won)
    settle_order(conn, order_id, settled_at=datetime.now(timezone.utc), pnl_cents=pnl)
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
