"""The Odds API client (ingestion/odds_api.py).

Free tier = 25 requests/day, so responses are cached hard — one pull per sport per day
(L4). Used as a no-vig fair-value cross-check against Kalshi prices (not as a model
feature — the model is odds-free). All calls wrapped in try/except (L9). The API key is
read from settings and never logged.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import httpx

from config import RAW_DIR, settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
WC_SPORT = "soccer_fifa_world_cup"


async def fetch_odds(
    sport: str = WC_SPORT, *, regions: str = "us", markets: str = "h2h"
) -> list[dict[str, Any]]:
    """Fetch decimal odds for a sport, caching one response per day (L4)."""
    cache = RAW_DIR / f"oddsapi_{sport}_{date.today().isoformat()}.json"
    if cache.exists():
        logger.info("Cache hit: %s", cache.name)
        return json.loads(cache.read_text())

    if not settings.the_odds_api_key:
        logger.error("Odds API key not configured; cannot fetch odds")
        return []

    params = {
        "apiKey": settings.the_odds_api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/sports/{sport}/odds", params=params, timeout=30.0
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:  # L9
        logger.error("Odds API fetch failed for %s: %s", sport, exc)
        return []

    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        logger.info("Odds API requests remaining today: %s", remaining)

    data = resp.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def novig_from_h2h(outcomes: list[dict[str, Any]]) -> dict[str, float]:
    """Convert a bookmaker's h2h decimal odds to no-vig implied probabilities.

    ``outcomes`` is the Odds API shape ``[{"name": "France", "price": 1.95}, ...]``.
    Returns ``{name: fair_probability}`` summing to 1.0, or ``{}`` on bad input.
    """
    raw: dict[str, float] = {}
    for outcome in outcomes:
        price = outcome.get("price")
        name = outcome.get("name")
        if name is None or not isinstance(price, (int, float)) or price <= 0:
            return {}
        raw[name] = 1.0 / float(price)
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {name: value / total for name, value in raw.items()}
