"""API-Football client (ingestion/api_football.py).

Live 2026 World Cup fixtures and injuries (league=1, season=2026). Free tier is 100
requests/day, so EVERY response is cached to data/raw/ once per day (L4). Auth via the
``x-apisports-key`` header (read from settings, never logged). All calls wrapped in
try/except (L9). Team names are normalized to the canonical (martj42) spelling so they
line up with our ELO ratings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

from config import RAW_DIR, settings
from features.teams import canonical

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"
WC_LEAGUE_ID = 1
WC_SEASON = 2026

# Fixture statuses that mean "not yet played" (worth generating signals for).
PREMATCH_STATUSES = frozenset({"NS", "TBD", "PST"})
# Statuses that mean the match is over and can be settled.
FINISHED_STATUSES = frozenset({"FT", "AET", "PEN"})


@dataclass(frozen=True)
class Fixture:
    """A normalized fixture with canonical team names."""

    fixture_id: int
    kickoff_utc: str  # ISO-8601 string from the API
    status: str  # short status code, e.g. "NS"
    home_team: str
    away_team: str
    round: str | None
    home_goals: int | None = None
    away_goals: int | None = None


async def _get(endpoint: str, params: dict[str, Any]) -> dict[str, Any] | None:
    if not settings.api_football_key:
        logger.error("API-Football key not configured; cannot fetch %s", endpoint)
        return None
    headers = {"x-apisports-key": settings.api_football_key}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=30.0
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:  # L9
        logger.error("API-Football GET %s failed: %s", endpoint, exc)
        return None
    return resp.json()


async def _cached_fetch(
    name: str, endpoint: str, params: dict[str, Any]
) -> list[dict[str, Any]]:
    cache = RAW_DIR / name
    if cache.exists():
        logger.info("Cache hit: %s", cache.name)
        return json.loads(cache.read_text()).get("response", [])
    data = await _get(endpoint, params)
    if data is None:
        return []
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data.get("response", [])


async def fetch_fixtures(
    *, league: int = WC_LEAGUE_ID, season: int = WC_SEASON
) -> list[dict[str, Any]]:
    """Fetch raw fixtures for the league-season (cached once per day)."""
    name = f"apifootball_fixtures_{league}_{season}_{date.today().isoformat()}.json"
    return await _cached_fetch(name, "/fixtures", {"league": league, "season": season})


async def fetch_injuries(
    *, league: int = WC_LEAGUE_ID, season: int = WC_SEASON
) -> list[dict[str, Any]]:
    """Fetch raw injuries for the league-season (cached once per day)."""
    name = f"apifootball_injuries_{league}_{season}_{date.today().isoformat()}.json"
    return await _cached_fetch(name, "/injuries", {"league": league, "season": season})


def parse_fixtures(raw: list[dict[str, Any]]) -> list[Fixture]:
    """Normalize the API-Football fixtures payload into :class:`Fixture` records."""
    fixtures: list[Fixture] = []
    for item in raw:
        try:
            fx = item["fixture"]
            teams = item["teams"]
            league = item.get("league", {})
            goals = item.get("goals") or {}
            fixtures.append(
                Fixture(
                    fixture_id=int(fx["id"]),
                    kickoff_utc=str(fx.get("date", "")),
                    status=str((fx.get("status") or {}).get("short", "")),
                    home_team=canonical(teams["home"]["name"]),
                    away_team=canonical(teams["away"]["name"]),
                    round=league.get("round"),
                    home_goals=_int_or_none(goals.get("home")),
                    away_goals=_int_or_none(goals.get("away")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:  # L9 — skip one bad row
            logger.warning("Skipping malformed fixture: %s", exc)
            continue
    return fixtures


def upcoming(fixtures: list[Fixture]) -> list[Fixture]:
    """Keep only fixtures that have not been played yet."""
    return [f for f in fixtures if f.status in PREMATCH_STATUSES]


def finished(fixtures: list[Fixture]) -> list[Fixture]:
    """Keep only fixtures that have finished (and so can be settled)."""
    return [f for f in fixtures if f.status in FINISHED_STATUSES]


def outcome(fixture: Fixture) -> str | None:
    """Match result as H/D/A from the goals, or None if not yet known."""
    if fixture.home_goals is None or fixture.away_goals is None:
        return None
    if fixture.home_goals > fixture.away_goals:
        return "H"
    if fixture.home_goals == fixture.away_goals:
        return "D"
    return "A"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
