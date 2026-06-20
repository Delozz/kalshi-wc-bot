"""Tests for the look-ahead guard (L1) — the backtester's most important safety net."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from backtest.lookahead_guard import LookAheadError, filter_data


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2018-06-01", "2018-06-10", "2018-06-20"],
            "value": [1, 2, 3],
        }
    )


def test_filters_rows_before_cutoff() -> None:
    cutoff = datetime(2018, 6, 15, tzinfo=timezone.utc)
    out = filter_data(_frame(), cutoff, strict=False)
    assert list(out["value"]) == [1, 2]


def test_raises_on_violation_when_strict() -> None:
    cutoff = datetime(2018, 6, 15, tzinfo=timezone.utc)
    with pytest.raises(LookAheadError):
        filter_data(_frame(), cutoff, strict=True)


def test_no_violation_passes_strict() -> None:
    cutoff = datetime(2018, 7, 1, tzinfo=timezone.utc)
    out = filter_data(_frame(), cutoff, strict=True)
    assert len(out) == 3


def test_missing_column_raises_keyerror() -> None:
    cutoff = datetime(2018, 6, 15, tzinfo=timezone.utc)
    with pytest.raises(KeyError):
        filter_data(_frame(), cutoff, timestamp_col="kickoff")


def test_naive_cutoff_treated_as_utc() -> None:
    cutoff = datetime(2018, 6, 15)  # naive — should be interpreted as UTC
    out = filter_data(_frame(), cutoff, strict=False)
    assert list(out["value"]) == [1, 2]


def test_unparseable_dates_dropped() -> None:
    df = pd.DataFrame({"date": ["2018-06-01", "not-a-date"], "value": [1, 2]})
    cutoff = datetime(2018, 7, 1, tzinfo=timezone.utc)
    out = filter_data(df, cutoff, strict=True)
    assert list(out["value"]) == [1]
