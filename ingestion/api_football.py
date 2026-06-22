"""API-Football client (ingestion/api_football.py).

Live 2026 World Cup fixtures, lineups, and injuries (league=1, season=2026). Pro tier:
7,500 requests/day / 300 per minute — responses still cached to data/raw/ (L4). Auth via the
``x-apisports-key`` header (read from settings, never logged). All calls wrapped in
try/except (L9). Team names are normalized to the canonical (martj42) spelling so they
line up with our ELO ratings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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


async def fetch_lineups(fixture_id: int) -> list[dict[str, Any]]:
    """Fetch confirmed lineups for a specific fixture (cached per fixture per day).

    Returns empty list if the lineup has not been announced yet (typically released
    ~1h before kickoff). Callers must handle the empty case gracefully.

    Unlike the generic cached fetch, an EMPTY response is never persisted: lineups are
    published progressively, so caching a pre-announcement empty would mask the real XI
    for the rest of the day. Only a non-empty lineup is written to the cache (L4).
    """
    name = f"apifootball_lineups_{fixture_id}_{date.today().isoformat()}.json"
    cache = RAW_DIR / name
    if cache.exists():
        cached = json.loads(cache.read_text()).get("response", [])
        if cached:  # trust the cache only once a real lineup has been stored
            logger.info("Cache hit: %s", cache.name)
            return cached
    data = await _get("/fixtures/lineups", {"fixture": fixture_id})
    response = (data or {}).get("response", [])
    if response:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    return response


# Bound the pagination loop so a malformed ``paging.total`` can never spin the quota.
_MAX_PLAYER_PAGES = 10


def _parse_player_ratings(items: list[dict[str, Any]]) -> dict[int, float]:
    """Map player id -> mean season rating from a ``/players`` response page.

    Each item may carry several ``statistics`` blocks (one per competition the player
    featured in that season); we average every numeric ``games.rating`` present. Players
    with no numeric rating yet (e.g. before any match is played) are omitted entirely so
    downstream strength deltas degrade to 0.0 rather than guessing.
    """
    ratings: dict[int, float] = {}
    for item in items:
        player = item.get("player") or {}
        pid = player.get("id")
        if pid is None:
            continue
        values: list[float] = []
        for stat in item.get("statistics") or []:
            raw = (stat.get("games") or {}).get("rating")
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if values:
            ratings[int(pid)] = sum(values) / len(values)
    return ratings


async def fetch_squad_ratings(
    team_id: int, *, season: int = WC_SEASON
) -> dict[int, float]:
    """Mean season player rating for a national-team squad, keyed by player id.

    Pulls ``/players?team=&season=`` across all pages and folds them through
    :func:`_parse_player_ratings`. Cached per team per day; like :func:`fetch_lineups`,
    an empty result is never persisted (ratings are appearance-derived, so they fill in
    as the tournament progresses — caching an early empty would freeze it for the day).
    """
    name = (
        f"apifootball_playerratings_{team_id}_{season}_{date.today().isoformat()}.json"
    )
    cache = RAW_DIR / name
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached:
            logger.info("Cache hit: %s", cache.name)
            return {int(k): float(v) for k, v in cached.items()}

    ratings: dict[int, float] = {}
    page = 1
    while page <= _MAX_PLAYER_PAGES:
        data = await _get("/players", {"team": team_id, "season": season, "page": page})
        if data is None:
            break
        ratings.update(_parse_player_ratings(data.get("response", [])))
        total = int((data.get("paging") or {}).get("total", 1) or 1)
        if page >= total:
            break
        page += 1

    if ratings:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({str(k): v for k, v in ratings.items()}))
    return ratings


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


def upcoming(
    fixtures: list[Fixture], *, min_hours_to_kickoff: float = 2.0
) -> list[Fixture]:
    """Keep only fixtures with at least ``min_hours_to_kickoff`` until kickoff.

    Checks both status (must be pre-match) AND wall-clock time. The daily fixture
    cache can carry stale NS entries for matches that are already in-progress or
    within Kalshi's pre-match no-order window (~2 h before kickoff). Excluding
    those prevents them from consuming position slots for bets that can't fill.
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=min_hours_to_kickoff)
    result = []
    for f in fixtures:
        if f.status not in PREMATCH_STATUSES:
            continue
        try:
            kickoff = datetime.fromisoformat(f.kickoff_utc.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if kickoff > cutoff:
            result.append(f)
    return result


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
