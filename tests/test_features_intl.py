"""Tests for the national-team feature builders (neutral-aware ELO, batch form, H2H)."""

from __future__ import annotations

import math

import pandas as pd

from features import elo, form, h2h


def test_expected_home_with_home_advantage() -> None:
    no_adv = elo.expected_home(1500.0, 1500.0)
    with_adv = elo.expected_home(1500.0, 1500.0, elo.HOME_ADVANTAGE)
    assert no_adv == 0.5
    assert with_adv > 0.5


def test_venue_adjustment_neutral_and_host() -> None:
    # Non-neutral: full home advantage.
    assert elo.venue_adjustment("A", "B", neutral=False) == elo.HOME_ADVANTAGE
    # Neutral, no host: nothing.
    assert elo.venue_adjustment("A", "B", neutral=True) == 0.0
    # Neutral WC, host is home team: positive bonus.
    assert (
        elo.venue_adjustment("Russia", "B", neutral=True, host="Russia")
        == elo.HOST_BONUS
    )
    # Neutral WC, host is away team: negative (advantage to the away host).
    assert (
        elo.venue_adjustment("A", "Russia", neutral=True, host="Russia")
        == -elo.HOST_BONUS
    )


def test_run_elo_neutral_flag_suppresses_home_advantage() -> None:
    neutral_match = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "fthg": [1],
            "ftag": [1],
            "ftr": ["D"],
            "neutral": [True],
        }
    )
    table = elo.run_elo(neutral_match)
    # A draw between equal teams at a neutral venue leaves both ratings unchanged.
    assert math.isclose(table.loc[0, "home_elo_pre"], elo.BASE_RATING)
    assert math.isclose(table.loc[0, "away_elo_pre"], elo.BASE_RATING)


def test_run_form_batch_matches_pointwise() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-01", "2018-01-08", "2018-01-15"]),
            "home_team": ["A", "C", "A"],
            "away_team": ["B", "A", "D"],
            "fthg": [3, 1, 0],
            "ftag": [0, 0, 0],
            "ftr": ["H", "H", "D"],
        }
    )
    table = form.run_form(matches)
    # The last row's home team (A) pre-match form should match the pointwise helper
    # computed over the two prior matches.
    expected = form.team_form(matches.iloc[:2], "A")["form_5"]
    assert math.isclose(table.loc[2, "form_5_home"], expected)


def test_run_h2h_causal_win_rate() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-06-01", "2018-01-01"]),
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "B", "B"],
            "fthg": [2, 0, 1],
            "ftag": [0, 0, 0],
            "ftr": ["H", "D", "H"],
        }
    )
    table = h2h.run_h2h(matches)
    # First meeting: no prior H2H -> NaN.
    assert math.isnan(table.loc[0, "h2h_home_win_rate"])
    # Third meeting: A won 1 of the 2 prior meetings (win, draw) -> 0.5.
    assert math.isclose(table.loc[2, "h2h_home_win_rate"], 0.5)
