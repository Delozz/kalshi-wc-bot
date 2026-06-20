"""football-data.co.uk ingestion — historical results + closing odds CSV downloader.

No API key required; static CSV files per league-season at
``https://www.football-data.co.uk/mmz4281/{season}/{league}.csv`` (season is a
4-digit code, e.g. ``1718`` for 2017/18; league is e.g. ``E0`` for the EPL).

Every download is cached to ``data/raw/`` (L4) and re-used on subsequent calls.
All network calls are wrapped in try/except so a single source failure never
crashes the pipeline (L9).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import pandas as pd

from config import RAW_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Normalized columns and their source candidates (closing odds preferred over opening).
_ODDS_CANDIDATES: dict[str, tuple[str, ...]] = {
    "b365h": ("B365CH", "B365H"),
    "b365d": ("B365CD", "B365D"),
    "b365a": ("B365CA", "B365A"),
    "psh": ("PSCH", "PSH", "PH"),
    "psd": ("PSCD", "PSD", "PD"),
    "psa": ("PSCA", "PSA", "PA"),
}


def _season_url(league: str, season: str) -> str:
    return f"{BASE_URL}/{season}/{league}.csv"


def _cache_path(league: str, season: str) -> Path:
    return RAW_DIR / f"footballdata_{league}_{season}.csv"


async def _download_one(
    client: httpx.AsyncClient, league: str, season: str
) -> Path | None:
    """Download a single league-season CSV, using the cache if present."""
    cache = _cache_path(league, season)
    if cache.exists():
        logger.info("Cache hit: %s", cache.name)
        return cache
    url = _season_url(league, season)
    try:
        resp = await client.get(url, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:  # L9 — never crash on one source
        logger.error("Download failed for %s/%s (%s): %s", season, league, url, exc)
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(resp.content)
    logger.info("Downloaded %s (%d bytes)", cache.name, len(resp.content))
    return cache


async def download_seasons(pairs: list[tuple[str, str]]) -> list[Path]:
    """Download many ``(league, season)`` CSVs concurrently; return cached paths."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        results = await asyncio.gather(
            *[_download_one(client, league, season) for league, season in pairs]
        )
    return [p for p in results if p is not None]


def download_seasons_sync(pairs: list[tuple[str, str]]) -> list[Path]:
    """Synchronous convenience wrapper around :func:`download_seasons`."""
    return asyncio.run(download_seasons(pairs))


def _first_present(raw: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for col in candidates:
        if col in raw.columns:
            return pd.to_numeric(raw[col], errors="coerce").astype("float64")
    return pd.Series([float("nan")] * len(raw), dtype="float64")


def _string_col(raw: pd.DataFrame, name: str) -> pd.Series:
    if name in raw.columns:
        return raw[name].astype("string")
    return pd.Series([pd.NA] * len(raw), dtype="string")


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a raw football-data.co.uk frame to the normalized match schema."""
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce"),
            "home_team": _string_col(raw, "HomeTeam"),
            "away_team": _string_col(raw, "AwayTeam"),
            "fthg": pd.to_numeric(raw.get("FTHG"), errors="coerce").astype("Int64"),
            "ftag": pd.to_numeric(raw.get("FTAG"), errors="coerce").astype("Int64"),
            "ftr": _string_col(raw, "FTR"),
        }
    )
    for norm_col, candidates in _ODDS_CANDIDATES.items():
        out[norm_col] = _first_present(raw, candidates)
    return out


def load_matches(paths: list[Path]) -> pd.DataFrame:
    """Load and normalize cached CSVs into a single date-sorted match frame."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            raw = pd.read_csv(path, encoding="latin-1")
        except (OSError, pd.errors.ParserError) as exc:  # L9
            logger.error("Failed to parse %s: %s", path, exc)
            continue
        frames.append(_normalize(raw))

    if not frames:
        logger.warning("No football-data CSVs loaded; returning empty frame")
        return _normalize(pd.DataFrame())

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date", "home_team", "away_team"])
    return df.sort_values("date").reset_index(drop=True)
