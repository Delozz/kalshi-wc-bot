"""Causal feature-matrix assembly for the odds-free national-team model (model/dataset.py).

Runs the causal batch feature builders (ELO, form, H2H) over the full chronological
frame and attaches a venue-adjusted ELO delta, a neutral-venue flag, and the outcome
label. Because every builder records pre-match values, the resulting table has no
look-ahead (L1).

The 2022 World Cup is the holdout (L2): callers must seal it upstream by loading
international results with ``max_date="2022-01-01"``. Nothing here touches 2022.
"""

from __future__ import annotations

import logging

import pandas as pd

from features import elo, form, h2h

logger = logging.getLogger(__name__)

# Odds-free feature set (team strength only — the model stays independent of the market).
FEATURE_COLUMNS: list[str] = [
    "elo_delta_adj",
    "home_elo_pre",
    "away_elo_pre",
    "form_5_home",
    "form_5_away",
    "form_10_home",
    "form_10_away",
    "goals_scored_5_home",
    "goals_conceded_5_home",
    "goals_scored_5_away",
    "goals_conceded_5_away",
    "h2h_home_win_rate",
    "h2h_goals_avg",
    "neutral_flag",
]
LABEL_MAP: dict[str, int] = {"H": 0, "D": 1, "A": 2}

# WC host nations by year (host gets a small ELO bonus at neutral WC venues).
WC_HOSTS: dict[int, set[str]] = {
    2018: {"Russia"},
    2026: {"United States", "Canada", "Mexico"},
}


def _host_for_match(
    home: str, away: str, year: int, hosts_by_year: dict[int, set[str]]
) -> str | None:
    hosts = hosts_by_year.get(year, set())
    if home in hosts:
        return home
    if away in hosts:
        return away
    return None


def build_feature_table(
    matches: pd.DataFrame, *, hosts_by_year: dict[int, set[str]] | None = None
) -> pd.DataFrame:
    """Build the causal feature table with venue-adjusted ELO delta and labels."""
    hosts_by_year = hosts_by_year or WC_HOSTS
    table = elo.run_elo(matches, use_tournament_k=True)
    table = form.run_form(table)
    table = h2h.run_h2h(table)

    adjusted: list[float] = []
    neutral_flags: list[float] = []
    for row in table.itertuples(index=False):
        is_wc = (
            isinstance(row.tournament, str) and "world cup" in row.tournament.lower()
        )
        is_neutral = bool(getattr(row, "neutral", False))
        host = (
            _host_for_match(row.home_team, row.away_team, row.date.year, hosts_by_year)
            if is_wc
            else None
        )
        bump = elo.venue_adjustment(
            row.home_team, row.away_team, neutral=is_neutral, host=host
        )
        adjusted.append(row.elo_delta + bump)
        neutral_flags.append(1.0 if is_neutral else 0.0)

    table["elo_delta_adj"] = adjusted
    table["neutral_flag"] = neutral_flags
    table["label"] = table["ftr"].map(LABEL_MAP)
    return table


def split_train_val(
    table: pd.DataFrame, *, val_year: int = 2018
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, validation).

    Validation = the ``val_year`` World Cup matches. Training = every match strictly
    before the first validation match, so the model never peeks at the WC it is scored
    on.
    """
    is_wc = table["tournament"].str.lower().str.contains("world cup", na=False)
    val = table[is_wc & (table["date"].dt.year == val_year)].copy()
    if len(val):
        cutoff = val["date"].min()
    else:
        cutoff = pd.Timestamp(f"{val_year}-06-01")
        logger.warning(
            "No %d WC matches found; using %s as train cutoff", val_year, cutoff.date()
        )
    train = table[table["date"] < cutoff].copy()
    logger.info(
        "Split: %d train rows, %d validation rows (%d WC)",
        len(train),
        len(val),
        val_year,
    )
    return train, val
