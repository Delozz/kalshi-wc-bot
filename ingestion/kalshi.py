"""Kalshi REST client (ingestion/kalshi.py).

Public market-data endpoints need no auth and read live World Cup prices and order
books. Authenticated endpoints (balance, positions, orders) use RSA-PSS request
signing: every signed request carries ``KALSHI-ACCESS-KEY`` (the key id),
``KALSHI-ACCESS-TIMESTAMP`` (ms), and ``KALSHI-ACCESS-SIGNATURE`` (base64 RSA-PSS of
``timestamp + METHOD + path``). The private key is read from settings (PEM text or a
file path) and never logged. Responses cached to ``data/raw/`` (L4); all network calls
wrapped in try/except (L9).

Order placement here is the low-level signed call; the demo-first guard and
orchestration live in execution/order_manager.py (L8).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from config import RAW_DIR, settings

logger = logging.getLogger(__name__)

PROD_HOST = "https://api.elections.kalshi.com"
DEMO_HOST = "https://demo-api.kalshi.co"
# Public market data always reads from production — demo does not mirror live series
# like KXWC26.  Signals must be computed against real prices regardless of which
# environment handles order execution.
PUBLIC_DATA_HOST = "https://external-api.kalshi.com"
PATH_PREFIX = "/trade-api/v2"

WC_SERIES_TICKER = "KXWC26"


def _host() -> str:
    """Trading host: demo for paper runs, prod after clean demo validation (L8)."""
    return DEMO_HOST if settings.kalshi_env == "demo" else PROD_HOST


def base_url() -> str:
    """Authenticated trading base URL (env-specific)."""
    return f"{_host()}{PATH_PREFIX}"


def public_base_url() -> str:
    """Public market-data base URL — always production (market data is real-world only)."""
    return f"{PUBLIC_DATA_HOST}{PATH_PREFIX}"


def _cache(name: str, payload: Any) -> None:
    path = RAW_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# --------------------------------------------------------------------------- auth


def load_private_key(secret: str | None = None) -> rsa.RSAPrivateKey | None:
    """Load the RSA private key from PEM text or a file path (settings by default)."""
    raw = settings.kalshi_api_secret if secret is None else secret
    if not raw:
        return None
    if "BEGIN" in raw:
        data = raw.encode("utf-8")
    else:
        path = Path(raw)
        if not path.exists():
            logger.error("Kalshi private key path not found: %s", path)
            return None
        data = path.read_bytes()
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError) as exc:
        logger.error("Failed to load Kalshi private key: %s", exc)
        return None
    if not isinstance(key, rsa.RSAPrivateKey):
        logger.error("Kalshi private key is not an RSA key")
        return None
    return key


def sign_message(private_key: rsa.RSAPrivateKey, message: str) -> str:
    """RSA-PSS (SHA-256, max salt) signature of ``message``, base64-encoded."""
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("ascii")


def signed_headers(
    method: str,
    path: str,
    *,
    private_key: rsa.RSAPrivateKey | None = None,
    key_id: str | None = None,
    timestamp_ms: int | None = None,
) -> dict[str, str]:
    """Build the three Kalshi auth headers for ``method`` and full ``path``."""
    pk = private_key or load_private_key()
    kid = key_id if key_id is not None else settings.kalshi_api_key
    if pk is None or not kid:
        raise RuntimeError(
            "Kalshi credentials missing: need KALSHI_API_KEY (key id) and "
            "KALSHI_API_SECRET (RSA private key PEM or path)."
        )
    ts = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
    message = ts + method.upper() + path
    return {
        "KALSHI-ACCESS-KEY": kid,
        "KALSHI-ACCESS-SIGNATURE": sign_message(pk, message),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# --------------------------------------------------------------- public endpoints


async def _get(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, Any] | None = None,
    *,
    use_public: bool = False,
) -> dict[str, Any] | None:
    url = f"{public_base_url() if use_public else base_url()}{endpoint}"
    try:
        resp = await client.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:  # L9
        logger.error("Kalshi GET %s failed: %s", endpoint, exc)
        return None
    return resp.json()


async def get_markets(
    series_ticker: str = WC_SERIES_TICKER, status: str = "open"
) -> list[dict[str, Any]]:
    """List markets for a series (default the 2026 WC). Always hits production (L4, L9).

    Market data is fetched from the production API regardless of KALSHI_ENV because
    the demo environment does not carry live series like KXWC26.
    """
    async with httpx.AsyncClient() as client:
        data = await _get(
            client,
            "/markets",
            {"series_ticker": series_ticker, "status": status},
            use_public=True,
        )
    if not data:
        return []
    _cache(f"kalshi_markets_{series_ticker}.json", data)
    return data.get("markets", [])


async def get_orderbook(ticker: str) -> dict[str, Any] | None:
    """Fetch the live order book for a market ticker (production public endpoint)."""
    async with httpx.AsyncClient() as client:
        data = await _get(client, f"/markets/{ticker}/orderbook", use_public=True)
    if data:
        _cache(f"kalshi_orderbook_{ticker}.json", data)
    return data


def implied_yes_price(market: dict[str, Any]) -> float | None:
    """YES ask price as a 0..1 probability (PRD 7.3: enter at the ask, not the mid).

    Prefers ``yes_ask_dollars`` (FixedPointDollars string, already 0–1).  Falls back to
    legacy ``yes_ask`` (integer cents) for any cached responses pre-dating the API
    migration to dollar strings.
    """
    if (ask_dollars := market.get("yes_ask_dollars")) is not None:
        return float(ask_dollars)
    ask_cents = market.get("yes_ask")
    if ask_cents is None:
        return None
    return float(ask_cents) / 100.0


# -------------------------------------------------------- authenticated endpoints


async def _authed_request(
    method: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        headers = signed_headers(method, f"{PATH_PREFIX}{endpoint}")
    except RuntimeError as exc:  # missing creds — never crash the caller (L9)
        logger.error("%s", exc)
        return None
    url = f"{base_url()}{endpoint}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:  # L9
        logger.error("Kalshi %s %s failed: %s", method, endpoint, exc)
        return None
    return resp.json()


async def get_balance() -> dict[str, Any] | None:
    """Authenticated: current portfolio balance (cents)."""
    return await _authed_request("GET", "/portfolio/balance")


async def get_positions() -> dict[str, Any] | None:
    """Authenticated: open positions."""
    return await _authed_request("GET", "/portfolio/positions")


async def create_order(
    *,
    ticker: str,
    action: str,
    side: str,
    count: int,
    yes_price_cents: int,
    order_type: str = "limit",
    client_order_id: str | None = None,
) -> dict[str, Any] | None:
    """Authenticated low-level order placement. Use order_manager for the demo guard (L8)."""
    body: dict[str, Any] = {
        "ticker": ticker,
        "action": action,
        "side": side,
        "count": count,
        "type": order_type,
        "yes_price": yes_price_cents,
    }
    if client_order_id is not None:
        body["client_order_id"] = client_order_id
    return await _authed_request("POST", "/portfolio/orders", json_body=body)


async def cancel_order(order_id: str) -> dict[str, Any] | None:
    """Authenticated: cancel a resting order."""
    return await _authed_request("DELETE", f"/portfolio/orders/{order_id}")


async def get_order(order_id: str) -> dict[str, Any] | None:
    """Authenticated: fetch a single order's current state."""
    return await _authed_request("GET", f"/portfolio/orders/{order_id}")
