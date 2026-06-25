"""Tests for the out-of-sample probability backtest (backtest/strategy_backtest.py).

These cover the two non-negotiable safety rails — the L2 holdout guard and the L1
look-ahead tripwire — without training a model or touching the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest import strategy_backtest as sb
from backtest.lookahead_guard import LookAheadError
from model.dataset import FEATURE_COLUMNS


def _frame(dates: list[str]) -> pd.DataFrame:
    """Minimal feature table: just enough columns for the look-ahead/_xy paths."""
    n = len(dates)
    data = {col: [0.0] * n for col in FEATURE_COLUMNS}
    data["date"] = pd.to_datetime(dates)
    data["label"] = [0] * n
    return pd.DataFrame(data)


def test_holdout_guard_refuses_2022_without_flag() -> None:
    # L2: any year >= 2022 must refuse before any data is even loaded.
    with pytest.raises(SystemExit, match="sacred holdout"):
        sb.run_probability_backtest(2022)


def test_holdout_guard_allows_with_explicit_flag(monkeypatch) -> None:
    # With allow_holdout=True the guard passes; stub the load so no real data is touched
    # and assert it proceeds past the guard into the (here empty) data path.
    monkeypatch.setattr(
        sb.international_results, "load", lambda *a, **k: pd.DataFrame()
    )
    with pytest.raises(SystemExit, match="No international results"):
        sb.run_probability_backtest(2022, allow_holdout=True)


def test_lookahead_tripwire_raises_on_leak() -> None:
    # A training row dated on/after the tournament's first kickoff is look-ahead (L1).
    train = _frame(["2018-06-20", "2018-07-01"])  # 07-01 is at/after val kickoff
    val = _frame(["2018-06-25"])
    with pytest.raises(LookAheadError):
        sb.evaluate_split(2018, train, val)


def test_evaluate_split_rejects_empty_val() -> None:
    train = _frame(["2017-01-01"])
    with pytest.raises(SystemExit, match="No 2018 World Cup"):
        sb.evaluate_split(2018, train, _frame([]))
