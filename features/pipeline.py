"""Feature pipeline (features/pipeline.py) — orchestrates all builders.

Every per-match build routes source data through the look-ahead guard (L1) before
computing anything. Dates are day-granular, so the guard excludes same-day matches
conservatively (no intraday leak): ``filter_data`` keeps rows strictly before the
cutoff date.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from backtest.lookahead_guard import filter_data
from features import elo, form, odds_features

logger = logging.getLogger(__name__)


def build_match_features(
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    cutoff: datetime,
    match_odds: Mapping[str, float],
    *,
    k: float = elo.K_GROUP,
) -> dict[str, float]:
    """Build the feature vector for one match using only data before ``cutoff`` (L1).

    The caller must pass ``history`` already restricted to rows before ``cutoff``.
    The guard runs in ``strict=True`` mode as a hard tripwire: if any row leaks at or
    after the cutoff it raises :class:`LookAheadError` rather than silently dropping
    it — look-ahead is fatal (L1), so the pipeline fails loudly instead of lying.
    """
    safe = filter_data(history, cutoff, timestamp_col="date", strict=True)

    ratings = elo.final_ratings(safe, k=k)
    delta = elo.elo_delta(ratings, home_team, away_team)

    home_form = form.team_form(safe, home_team)
    away_form = form.team_form(safe, away_team)

    implied_h, implied_d, implied_a = odds_features.pinnacle_novig(
        match_odds.get("psh", float("nan")),
        match_odds.get("psd", float("nan")),
        match_odds.get("psa", float("nan")),
    )

    return {
        "elo_delta": delta,
        "form_5_home": home_form["form_5"],
        "form_5_away": away_form["form_5"],
        "form_10_home": home_form["form_10"],
        "form_10_away": away_form["form_10"],
        "goals_scored_5_home": home_form["goals_scored_5"],
        "goals_conceded_5_home": home_form["goals_conceded_5"],
        "pinnacle_implied_home": implied_h,
        "pinnacle_implied_draw": implied_d,
        "pinnacle_implied_away": implied_a,
    }


def build_feature_table(
    matches: pd.DataFrame, *, k: float = elo.K_GROUP
) -> pd.DataFrame:
    """Batch causal feature table for training.

    ELO pre-match ratings and odds implied probabilities are causal by construction,
    so the whole frame can be processed in one pass without look-ahead.
    """
    table = elo.run_elo(matches, k=k)
    table = odds_features.add_odds_features(table)
    return table
