"""Tests for the model dataset assembly (feature columns, labels, venue adjustment)."""

from __future__ import annotations

import pandas as pd

from features import elo
from model import dataset


def _synthetic_internationals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2014-06-01", "2015-06-01", "2016-06-01", "2018-06-14"]
            ),
            "home_team": ["Russia", "Russia", "Brazil", "Russia"],
            "away_team": ["Brazil", "Brazil", "Russia", "Brazil"],
            "fthg": [1, 2, 0, 3],
            "ftag": [1, 0, 2, 1],
            "ftr": ["D", "H", "A", "H"],
            "tournament": ["Friendly", "Friendly", "Friendly", "FIFA World Cup"],
            "neutral": [False, False, False, True],
        }
    )


def test_build_feature_table_has_all_feature_columns() -> None:
    table = dataset.build_feature_table(_synthetic_internationals())
    for column in dataset.FEATURE_COLUMNS:
        assert column in table.columns, f"missing feature column: {column}"
    assert "label" in table.columns


def test_labels_map_outcomes() -> None:
    table = dataset.build_feature_table(_synthetic_internationals())
    # Outcomes were H/D/A/H -> labels 0/1/2 via LABEL_MAP.
    assert list(table["label"]) == [1, 0, 2, 0]


def test_neutral_flag_set_for_world_cup_row() -> None:
    table = dataset.build_feature_table(_synthetic_internationals())
    wc_row = table[table["tournament"] == "FIFA World Cup"].iloc[0]
    assert wc_row["neutral_flag"] == 1.0


def test_host_bonus_applied_at_neutral_wc() -> None:
    table = dataset.build_feature_table(_synthetic_internationals())
    wc_row = table[table["tournament"] == "FIFA World Cup"].iloc[0]
    # Russia (2018 host) is the home team at a neutral WC venue, so the venue-adjusted
    # delta must exceed the raw ELO delta by exactly the host bonus.
    assert wc_row["elo_delta_adj"] == wc_row["elo_delta"] + elo.HOST_BONUS


def test_split_train_val_isolates_world_cup() -> None:
    table = dataset.build_feature_table(_synthetic_internationals())
    train, val = dataset.split_train_val(table, val_year=2018)
    assert len(val) == 1
    assert (val["tournament"] == "FIFA World Cup").all()
    # Training rows are strictly before the WC match.
    assert (train["date"] < val["date"].min()).all()
