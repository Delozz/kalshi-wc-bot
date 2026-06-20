"""Head-to-head features (features/h2h.py).

Causal: only prior meetings of the same pair are used. ``run_h2h`` is a batch builder
that records each row's pre-match H2H stats before updating the pair history.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WINDOW: int = 5


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def run_h2h(matches: pd.DataFrame, *, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Batch causal H2H: ``h2h_home_win_rate`` and ``h2h_goals_avg`` per row.

    ``h2h_home_win_rate`` is from the current home team's perspective over the last
    ``window`` meetings; ``h2h_goals_avg`` is the average total goals in those meetings.
    Pairs with no prior meetings yield NaN.
    """
    ordered = matches.sort_values("date").reset_index(drop=True)
    history: dict[tuple[str, str], list[tuple[str, str, int, int]]] = defaultdict(list)

    win_rate: list[float] = []
    goals_avg: list[float] = []
    for row in ordered.itertuples(index=False):
        key = _pair_key(row.home_team, row.away_team)
        recent = history[key][-window:]
        wins = 0
        total_goals = 0
        counted = 0
        for prev_home, _prev_away, fh, fa in recent:
            counted += 1
            total_goals += fh + fa
            home_won_then = fh > fa if prev_home == row.home_team else fa > fh
            if home_won_then:
                wins += 1
        win_rate.append(wins / counted if counted else float("nan"))
        goals_avg.append(total_goals / counted if counted else float("nan"))

        if not (pd.isna(row.fthg) or pd.isna(row.ftag)):
            history[key].append(
                (row.home_team, row.away_team, int(row.fthg), int(row.ftag))
            )

    ordered["h2h_home_win_rate"] = win_rate
    ordered["h2h_goals_avg"] = goals_avg
    return ordered


def h2h_for_match(
    matches: pd.DataFrame, home: str, away: str, *, window: int = DEFAULT_WINDOW
) -> dict[str, float]:
    """Point-in-time H2H for one pairing from already-known matches.

    Returns ``{"win_rate": .., "goals_avg": ..}`` from the home team's perspective over
    the most recent ``window`` meetings; NaN when the pair has never met. ``matches`` is
    assumed to already exclude anything at/after the fixture (for a future fixture, all
    known results precede it).
    """
    mask = ((matches["home_team"] == home) & (matches["away_team"] == away)) | (
        (matches["home_team"] == away) & (matches["away_team"] == home)
    )
    recent = matches[mask].sort_values("date").tail(window)

    wins = 0
    total_goals = 0
    counted = 0
    for row in recent.itertuples(index=False):
        if pd.isna(row.fthg) or pd.isna(row.ftag):
            continue
        counted += 1
        total_goals += int(row.fthg) + int(row.ftag)
        home_won = (
            (row.fthg > row.ftag) if row.home_team == home else (row.ftag > row.fthg)
        )
        if home_won:
            wins += 1
    return {
        "win_rate": wins / counted if counted else float("nan"),
        "goals_avg": total_goals / counted if counted else float("nan"),
    }
