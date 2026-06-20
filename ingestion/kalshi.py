"""Kalshi REST client (ingestion/kalshi.py).

Public market-data endpoints need no auth and are used to read live World Cup prices
and order books. Authenticated trading (balance, positions, orders) requires signed
requests and is wired in the execution phase — and only after a full demo paper run
(L8: never send prod orders before a clean demo run). Responses are cached to
``data/raw/`` (L4); all network calls are wrapped in try/except (L9).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from config import RAW_DIR, settings

logger = logging.getLogger(__name__)

PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"

WC_SERIES_TICKER = "KXWC26"


def base_url() -> str:
    """Pick the API host from KALSHI_ENV (demo by default until validated, L8)."""
    return DEMO_BASE if settings.kalshi_env == "demo" else PROD_BASE


def _cache(name: str, payload: Any) -> None:
    path = RAW_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


async def _get(
    client: httpx.AsyncClient, path: str, params: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    url = f"{base_url()}{path}"
    try:
        resp = await client.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:  # L9
        logger.error("Kalshi GET %s failed: %s", path, exc)
        return None
    return resp.json()


async def get_markets(
    series_ticker: str = WC_SERIES_TICKER, status: str = "open"
) -> list[dict[str, Any]]:
    """List markets for a series (default the 2026 WC). Empty list on failure (L9)."""
    async with httpx.AsyncClient() as client:
        data = await _get(
            client, "/markets", {"series_ticker": series_ticker, "status": status}
        )
    if not data:
        return []
    _cache(f"kalshi_markets_{series_ticker}.json", data)
    return data.get("markets", [])


async def get_orderbook(ticker: str) -> dict[str, Any] | None:
    """Fetch the live order book for a market ticker."""
    async with httpx.AsyncClient() as client:
        data = await _get(client, f"/markets/{ticker}/orderbook")
    if data:
        _cache(f"kalshi_orderbook_{ticker}.json", data)
    return data


def implied_yes_price(market: dict[str, Any]) -> float | None:
    """YES ask price as a 0..1 probability (PRD 7.3: enter at the ask, not the mid)."""
    ask = market.get("yes_ask")
    if ask is None:
        return None
    return float(ask) / 100.0


def _auth_not_wired() -> None:
    raise NotImplementedError(
        "Authenticated Kalshi requests (request signing) are implemented in the "
        "execution phase (execution/order_manager.py). Trading must complete a full "
        "demo paper run before any prod order (L8)."
    )


def get_balance() -> None:
    """Authenticated — not wired yet (execution phase, demo-first per L8)."""
    _auth_not_wired()


def get_positions() -> None:
    """Authenticated — not wired yet (execution phase, demo-first per L8)."""
    _auth_not_wired()


def place_order(*_args: Any, **_kwargs: Any) -> None:
    """Authenticated — not wired yet (execution phase, demo-first per L8)."""
    _auth_not_wired()
