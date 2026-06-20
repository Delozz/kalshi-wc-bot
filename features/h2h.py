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
