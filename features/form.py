"""Rolling form windows (features/form.py).

Causal: only matches strictly before the target are used. Points are 3/1/0 for
win/draw/loss from the team's perspective.

- ``team_form``: point-in-time form for one team from already cutoff-filtered matches.
- ``run_form``: batch causal builder that records pre-match form for every row in a
  single chronological pass (records before updating, so no look-ahead).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def team_form(
    matches: pd.DataFrame, team: str, *, windows: tuple[int, ...] = DEFAULT_WINDOWS
) -> dict[str, float]:
    """Form metrics for one team from already cutoff-filtered matches.

    Returns ``form_{w}`` (points per game over last ``w`` games) for each window plus
    ``goals_scored_5`` / ``goals_conceded_5``. Missing history yields NaN.
    """
    rows = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    rows = rows.sort_values("date")

    points: list[float] = []
    goals_for: list[float] = []
    goals_against: list[float] = []
    for row in rows.itertuples(index=False):
        is_home = row.home_team == team
        gf = row.fthg if is_home else row.ftag
        ga = row.ftag if is_home else row.fthg
        if pd.isna(gf) or pd.isna(ga):
            continue
        gf_i, ga_i = int(gf), int(ga)
        points.append(3.0 if gf_i > ga_i else 1.0 if gf_i == ga_i else 0.0)
        goals_for.append(float(gf_i))
        goals_against.append(float(ga_i))

    out: dict[str, float] = {f"form_{w}": _mean(points[-w:]) for w in windows}
    out["goals_scored_5"] = _mean(goals_for[-5:])
    out["goals_conceded_5"] = _mean(goals_against[-5:])
    return out


def _team_metrics(
    history: list[tuple[float, float, float]], windows: tuple[int, ...]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for w in windows:
        pts = [entry[0] for entry in history[-w:]]
        out[f"form_{w}"] = _mean(pts)
    out["goals_scored_5"] = _mean([entry[1] for entry in history[-5:]])
    out["goals_conceded_5"] = _mean([entry[2] for entry in history[-5:]])
    return out


def run_form(
    matches: pd.DataFrame, *, windows: tuple[int, ...] = DEFAULT_WINDOWS
) -> pd.DataFrame:
    """Batch causal form: per-row pre-match form for the home and away teams.

    Returns the date-sorted frame with ``form_{w}_home`` / ``form_{w}_away`` for each
    window plus ``goals_scored_5_*`` / ``goals_conceded_5_*``.
    """
    ordered = matches.sort_values("date").reset_index(drop=True)
    history: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    columns: dict[str, list[float]] = {}
    for side in ("home", "away"):
        for w in windows:
            columns[f"form_{w}_{side}"] = []
        columns[f"goals_scored_5_{side}"] = []
        columns[f"goals_conceded_5_{side}"] = []

    for row in ordered.itertuples(index=False):
        home_metrics = _team_metrics(history[row.home_team], windows)
        away_metrics = _team_metrics(history[row.away_team], windows)
        for w in windows:
            columns[f"form_{w}_home"].append(home_metrics[f"form_{w}"])
            columns[f"form_{w}_away"].append(away_metrics[f"form_{w}"])
        columns["goals_scored_5_home"].append(home_metrics["goals_scored_5"])
        columns["goals_conceded_5_home"].append(home_metrics["goals_conceded_5"])
        columns["goals_scored_5_away"].append(away_metrics["goals_scored_5"])
        columns["goals_conceded_5_away"].append(away_metrics["goals_conceded_5"])

        if pd.isna(row.fthg) or pd.isna(row.ftag):
            continue
        gh, ga = int(row.fthg), int(row.ftag)
        history[row.home_team].append(
            (3.0 if gh > ga else 1.0 if gh == ga else 0.0, float(gh), float(ga))
        )
        history[row.away_team].append(
            (3.0 if ga > gh else 1.0 if ga == gh else 0.0, float(ga), float(gh))
        )

    for name, values in columns.items():
        ordered[name] = values
    return ordered
