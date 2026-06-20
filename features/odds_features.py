"""Implied probability from odds (features/odds_features.py).

Pinnacle no-vig fair probabilities — the sharpest single feature (PRD section 5.3).
``raw = 1 / decimal_odds``; divide by the overround to strip the bookmaker margin.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)

_NAN3: tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))


def _invalid(value: float) -> bool:
    return (
        value is None or (isinstance(value, float) and math.isnan(value)) or value <= 0
    )


def pinnacle_novig(psh: float, psd: float, psa: float) -> tuple[float, float, float]:
    """Convert Pinnacle decimal odds to no-vig (home, draw, away) probabilities."""
    if any(_invalid(o) for o in (psh, psd, psa)):
        return _NAN3
    raw_h, raw_d, raw_a = 1.0 / psh, 1.0 / psd, 1.0 / psa
    overround = raw_h + raw_d + raw_a
    if overround <= 0:
        return _NAN3
    return (raw_h / overround, raw_d / overround, raw_a / overround)


def add_odds_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Vectorized no-vig implied probabilities for a whole match frame."""
    out = matches.copy()
    raw_h = 1.0 / out["psh"]
    raw_d = 1.0 / out["psd"]
    raw_a = 1.0 / out["psa"]
    overround = raw_h + raw_d + raw_a
    out["pinnacle_implied_home"] = raw_h / overround
    out["pinnacle_implied_draw"] = raw_d / overround
    out["pinnacle_implied_away"] = raw_a / overround
    return out
