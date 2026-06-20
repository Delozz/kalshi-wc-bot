"""Order manager (execution/order_manager.py).

Turns a sized :class:`Signal` into a Kalshi limit order, with two hard safety layers:

1. ``dry_run`` defaults to True — nothing is sent unless the caller explicitly opts in.
2. L8 demo-first: a prod order is refused unless ``KALSHI_ENV=demo`` OR the operator has
   set ``KALSHI_ALLOW_PROD_ORDERS=1`` after a clean demo paper run.

Orders are placed at the ask (PRD 7.3, never market orders). Every attempt can be logged
to the ``orders`` table.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from ingestion import kalshi
from schemas import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderRequest:
    """Concrete order parameters derived from a signal."""

    ticker: str
    action: str  # "buy"
    side: str  # "yes" / "no"
    count: int
    limit_price_cents: int


def order_from_signal(signal: Signal) -> OrderRequest | None:
    """Convert a signal into a whole-contract order at the implied price; None if unfillable."""
    price_cents = round(signal["market_implied"] * 100)
    if price_cents <= 0 or price_cents >= 100:
        logger.warning(
            "Signal %s has out-of-range price %d", signal["market_ticker"], price_cents
        )
        return None
    count = int(signal["bet_size_cents"] // price_cents)
    if count < 1:
        logger.info(
            "Signal %s sized below one contract; skipping", signal["market_ticker"]
        )
        return None
    return OrderRequest(
        ticker=signal["market_ticker"],
        action="buy",
        side=str(signal["side"]).lower(),
        count=count,
        limit_price_cents=price_cents,
    )


def prod_orders_allowed() -> bool:
    """L8: live prod orders require an explicit opt-in after a clean demo run."""
    return os.getenv("KALSHI_ALLOW_PROD_ORDERS") == "1"


async def place_order(
    signal: Signal, *, dry_run: bool = True
) -> dict[str, object] | None:
    """Place (or simulate) the order for a signal. Dry-run by default."""
    request = order_from_signal(signal)
    if request is None:
        return None

    if dry_run:
        logger.info("DRY RUN — would place %s", request)
        return {
            "status": "dry_run",
            "request": request,
            "placed_at": datetime.now(timezone.utc),
        }

    from config import settings

    if settings.kalshi_env == "prod" and not prod_orders_allowed():
        logger.error(
            "Refusing prod order for %s (L8: demo-first). Complete a demo paper run, "
            "then set KALSHI_ALLOW_PROD_ORDERS=1.",
            request.ticker,
        )
        return None

    logger.info(
        "Placing %s order: %d x %s @ %dc on %s",
        settings.kalshi_env,
        request.count,
        request.side,
        request.limit_price_cents,
        request.ticker,
    )
    return await kalshi.create_order(
        ticker=request.ticker,
        action=request.action,
        side=request.side,
        count=request.count,
        yes_price_cents=request.limit_price_cents,
    )
