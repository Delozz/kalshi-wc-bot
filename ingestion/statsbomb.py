"""StatsBomb open-data ingestion via statsbombpy.

WARNING (L2 — Holdout Is Sacred): The only World Cup in StatsBomb's free open data
is 2022, which is our SACRED HOLDOUT set. Do NOT pull or use 2022 WC data during
development or hyperparameter tuning. This module exists for the live/eval pipeline.
``competition_id`` and ``season_id`` are explicit parameters so nothing touches the
holdout by accident.

All network calls are wrapped in try/except so a single source failure never
crashes the pipeline (L9). Responses are cached to ``data/raw/`` (L4).
"""

from __future__ import annotations

import logging

import pandas as pd

from config import RAW_DIR

logger = logging.getLogger(__name__)

# (competition_id, season_id) for the 2022 men's World Cup — HOLDOUT, dev use forbidden (L2).
WC_2022_HOLDOUT: tuple[int, int] = (43, 106)


def fetch_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Fetch the StatsBomb match list for a competition-season, with caching.

    Returns an empty DataFrame on any failure (missing dependency or network error)
    rather than raising, so callers in the scheduler keep running (L9).
    """
    cache = RAW_DIR / f"statsbomb_matches_{competition_id}_{season_id}.json"
    if cache.exists():
        logger.info("Cache hit: %s", cache.name)
        return pd.read_json(cache)

    try:
        from statsbombpy import sb  # imported lazily; heavy optional dependency
    except ImportError:
        logger.error("statsbombpy not installed; cannot fetch StatsBomb data")
        return pd.DataFrame()

    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
    except Exception as exc:  # noqa: BLE001 — statsbombpy raises broad errors; L9
        logger.error(
            "StatsBomb matches fetch failed (%s/%s): %s",
            competition_id,
            season_id,
            exc,
        )
        return pd.DataFrame()

    cache.parent.mkdir(parents=True, exist_ok=True)
    matches.to_json(cache)
    logger.info("Cached %d StatsBomb matches to %s", len(matches), cache.name)
    return matches
