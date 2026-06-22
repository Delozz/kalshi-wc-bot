"""martj42 international results ingestion — national-team match history.

Source: https://raw.githubusercontent.com/martj42/international_results/master/results.csv
Men's full internationals from 1872 to the present. Free, no API key. The response is
cached to ``data/raw/`` (L4) and the network call is wrapped in try/except (L9).

Two columns matter beyond the basic result schema:
- ``neutral`` — WC matches are played at neutral venues (except the host), so ELO must
  be neutral-aware or it is structurally biased on internationals.
- ``tournament`` — drives the ELO K-factor (friendly < qualifier < major final).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from config import RAW_DIR

logger = logging.getLogger(__name__)

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
_CACHE_NAME = "martj42_results.csv"

_TRUE_STRINGS = {"true", "1", "yes", "t"}


def _cache_path() -> Path:
    return RAW_DIR / _CACHE_NAME


async def _download(client: httpx.AsyncClient) -> Path | None:
    cache = _cache_path()
    if cache.exists():
        logger.info("Cache hit: %s", cache.name)
        return cache
    try:
        resp = await client.get(RESULTS_URL, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:  # L9
        logger.error("martj42 results download failed (%s): %s", RESULTS_URL, exc)
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(resp.content)
    logger.info("Downloaded %s (%d bytes)", cache.name, len(resp.content))
    return cache


def download_sync() -> Path | None:
    """Download (or reuse the cache of) the martj42 results CSV."""

    async def _run() -> Path | None:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await _download(client)

    return asyncio.run(_run())


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return bool(value)


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], errors="coerce", utc=False),
            "home_team": raw["home_team"].astype("string"),
            "away_team": raw["away_team"].astype("string"),
            "fthg": pd.to_numeric(raw["home_score"], errors="coerce").astype("Int64"),
            "ftag": pd.to_numeric(raw["away_score"], errors="coerce").astype("Int64"),
            "tournament": raw["tournament"].astype("string"),
            "country": raw["country"].astype("string"),
            "neutral": raw["neutral"].map(_to_bool).astype("boolean"),
        }
    )
    # Drop incomplete rows BEFORE deriving the result, so the comparison runs on clean
    # integers and never sees <NA> (which would break np.select).
    out = out.dropna(subset=["date", "home_team", "away_team", "fthg", "ftag"])
    home = out["fthg"].astype(int).to_numpy()
    away = out["ftag"].astype(int).to_numpy()
    out["ftr"] = pd.Series(
        np.select([home > away, home == away], ["H", "D"], default="A"),
        index=out.index,
        dtype="string",
    )
    return out.sort_values("date").reset_index(drop=True)


async def load_async(*, max_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Async variant of ``load()`` for use inside async contexts (e.g. run_live).

    Calls ``_download`` directly so it never nests ``asyncio.run()`` inside a running
    event loop, which would raise RuntimeError.
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        path = await _download(client)
    if path is None:
        logger.error("No international results available; returning empty frame")
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:  # L9
        logger.error("Failed to parse %s: %s", path, exc)
        return pd.DataFrame()
    df = _normalize(raw)
    if max_date is not None:
        cutoff = pd.Timestamp(max_date)
        df = df[df["date"] < cutoff].reset_index(drop=True)
        logger.info("Filtered to %d matches before %s", len(df), cutoff.date())
    return df


def load(*, max_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Download, cache, and normalize the international results.

    Args:
        max_date: if given, keep only matches strictly before this date. Use this to
            SEAL the 2022 holdout (L2): pass ``"2022-01-01"`` to guarantee no 2022 WC
            data is ever loaded during development.
    """
    path = download_sync()
    if path is None:
        logger.error("No international results available; returning empty frame")
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:  # L9
        logger.error("Failed to parse %s: %s", path, exc)
        return pd.DataFrame()

    df = _normalize(raw)
    if max_date is not None:
        cutoff = pd.Timestamp(max_date)
        df = df[df["date"] < cutoff].reset_index(drop=True)
        logger.info("Filtered to %d matches before %s", len(df), cutoff.date())
    return df
