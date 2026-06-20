"""Unit tests for the Phase 1 feature builders (elo, form, odds)."""

from __future__ import annotations

import math

import pandas as pd

from features import elo, form, odds_features


def test_expected_home_is_half_for_equal_ratings() -> None:
    assert elo.expected_home(1500.0, 1500.0) == 0.5


def test_expected_home_monotonic_in_rating_gap() -> None:
    assert elo.expected_home(1700.0, 1500.0) > elo.expected_home(1500.0, 1500.0)


def test_run_elo_records_causal_pre_match_ratings() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-01", "2018-01-08"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "fthg": [2, 0],
            "ftag": [0, 0],
            "ftr": ["H", "D"],
        }
    )
    table = elo.run_elo(matches)
    # First match: both teams start at base, so delta is 0.
    assert table.loc[0, "elo_delta"] == 0.0
    # Second match: A won the first, so A should be rated above B pre-match.
    assert table.loc[1, "elo_delta"] > 0.0


def test_team_form_points_per_game() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-01", "2018-01-08", "2018-01-15"]),
            "home_team": ["A", "C", "A"],
            "away_team": ["B", "A", "D"],
            "fthg": [3, 1, 0],  # A wins, A loses (away), A draws
            "ftag": [0, 0, 0],
            "ftr": ["H", "H", "D"],
        }
    )
    result = form.team_form(matches, "A")
    # A: win (3) + loss (0) + draw (1) = 4 points over 3 games => 4/3 ppg.
    assert math.isclose(result["form_5"], 4.0 / 3.0)


def test_pinnacle_novig_sums_to_one() -> None:
    home, draw, away = odds_features.pinnacle_novig(2.0, 3.5, 4.0)
    assert math.isclose(home + draw + away, 1.0)


def test_pinnacle_novig_handles_invalid_odds() -> None:
    home, draw, away = odds_features.pinnacle_novig(0.0, 3.5, 4.0)
    assert math.isnan(home) and math.isnan(draw) and math.isnan(away)
