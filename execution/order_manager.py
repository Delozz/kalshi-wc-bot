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
from typing import Any

from ingestion import kalshi
from schemas import Order, Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderRequest:
    """Concrete order parameters derived from a signal."""

    ticker: str
    side: str  # V2 API: "bid" (buy YES) or "ask" (sell YES)
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
    # V2 API uses "bid"/"ask" (YES-leg perspective): buying YES = "bid"
    side_v2 = "bid" if str(signal["side"]).upper() == "YES" else "ask"
    return OrderRequest(
        ticker=signal["market_ticker"],
        side=side_v2,
        count=count,
        limit_price_cents=price_cents,
    )


def prod_orders_allowed() -> bool:
    """L8: live prod orders require an explicit opt-in after a clean demo run."""
    return os.getenv("KALSHI_ALLOW_PROD_ORDERS") == "1"


def _extract_order_id(response: object) -> str | None:
    """Best-effort pull of the order id from a Kalshi create-order response.

    V2 (/portfolio/events/orders) returns a flat ``{"order_id": "..."}`` dict.
    The legacy shape nested it under ``{"order": {"order_id": "..."}}``.
    """
    if not isinstance(response, dict):
        return None
    if response.get("order_id"):
        return str(response["order_id"])
    order = response.get("order")
    if isinstance(order, dict) and order.get("order_id"):
        return str(order["order_id"])
    return None


async def place_order(signal: Signal, *, dry_run: bool = True) -> dict[str, Any] | None:
    """Place (or simulate) the order for a signal. Dry-run by default.

    Returns ``{"status", "request", "order_id", "response"}`` so the caller can log the
    order row. ``status`` is "dry_run" or "placed"; ``order_id`` is None for a dry run.
    """
    request = order_from_signal(signal)
    if request is None:
        return None

    if dry_run:
        logger.info("DRY RUN — would place %s", request)
        return {
            "status": "dry_run",
            "request": request,
            "order_id": None,
            "response": None,
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
    response = await kalshi.create_order(
        ticker=request.ticker,
        side=request.side,
        count=request.count,
        yes_price_cents=request.limit_price_cents,
    )
    return {
        "status": "placed",
        "request": request,
        "order_id": _extract_order_id(response),
        "response": response,
    }


def build_order_row(
    *,
    order_id: str,
    signal_id: int,
    request: OrderRequest,
    status: str = "pending",
    filled_price: float | None = None,
) -> Order:
    """Build an :class:`Order` row for the database from a placed request."""
    return Order(
        id=order_id,
        signal_id=signal_id,
        status=status,  # type: ignore[typeddict-item]
        limit_price=request.limit_price_cents / 100.0,
        contracts=request.count,
        filled_price=filled_price,
        placed_at=datetime.now(timezone.utc),
        settled_at=None,
        pnl_cents=None,
    )


_FILLED_STATUSES = frozenset({"filled", "executed"})
_CANCELED_STATUSES = frozenset({"canceled", "cancelled"})


async def _kalshi_status(order_id: str) -> str:
    resp = await kalshi.get_order(order_id)
    if not resp:
        return "unknown"
    return str((resp.get("order") or {}).get("status", ""))


async def await_fill(
    order_id: str,
    *,
    timeout_s: int = 300,
    interval_s: int = 30,
    status_fn: Any = None,
    cancel_fn: Any = None,
    sleeper: Any = None,
) -> str:
    """Poll an order until filled/canceled; cancel and return "timeout" if unfilled.

    ``status_fn``/``cancel_fn``/``sleeper`` are injectable for testing (defaults hit the
    live Kalshi API and asyncio.sleep). Poll every ``interval_s`` up to ``timeout_s``
    (PRD 9.1: poll every 30s, cancel after 5 minutes).
    """
    import asyncio

    status_fn = status_fn or _kalshi_status
    cancel_fn = cancel_fn or kalshi.cancel_order
    sleeper = sleeper or asyncio.sleep

    waited = 0
    while waited < timeout_s:
        status = await status_fn(order_id)
        if status in _FILLED_STATUSES:
            return "filled"
        if status in _CANCELED_STATUSES:
            return "canceled"
        await sleeper(interval_s)
        waited += interval_s

    logger.warning("Order %s unfilled after %ds; cancelling", order_id, timeout_s)
    await cancel_fn(order_id)
    return "timeout"
