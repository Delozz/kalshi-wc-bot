"""Integration tests for signal generation (strategy/signal_gen.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import elo
from ingestion.api_football import Fixture
from model import dataset
from model.baseline import build_baseline
from strategy import signal_gen


def _toy_bundle() -> dict:
    cols = dataset.FEATURE_COLUMNS
    rng = np.random.RandomState(0)
    x = rng.randn(90, len(cols))
    y = np.array([0, 1, 2] * 30)
    estimator = build_baseline().fit(x, y)
    return {
        "feature_columns": cols,
        "selected": "baseline",
        "baseline": estimator,
        "model": estimator,
    }


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-01", "2024-09-01"]),
            "home_team": ["Brazil", "Serbia"],
            "away_team": ["Serbia", "Brazil"],
            "fthg": [2, 0],
            "ftag": [0, 1],
            "ftr": ["H", "A"],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [False, False],
        }
    )


def _fixture() -> Fixture:
    return Fixture(
        fixture_id=1,
        kickoff_utc="2026-06-20T18:00:00+00:00",
        status="NS",
        home_team="Brazil",
        away_team="Serbia",
        round="Group G",
    )


def _markets(yes_ask: int) -> list[dict]:
    return [
        {
            "ticker": "KXWC26-BRA",
            "title": "Brazil vs Serbia",
            "yes_sub_title": "Brazil",
            "yes_ask": yes_ask,
            "open_interest": 10000,
        }
    ]


def _ratings(history: pd.DataFrame) -> dict[str, float]:
    return elo.final_ratings(history, use_tournament_k=True)


def test_generates_signal_when_edge_exists() -> None:
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle=_toy_bundle(),
        markets=_markets(yes_ask=5),  # 5c price -> large edge vs any model prob
        bankroll_cents=20000,
    )
    assert len(signals) == 1
    assert signals[0]["market_ticker"] == "KXWC26-BRA"
    assert signals[0]["side"] == "YES"
    # Hard cap: never more than 5% of $200 bankroll = 1000 cents.
    assert 0 < signals[0]["bet_size_cents"] <= 1000


def test_no_signal_when_no_market_matches() -> None:
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle=_toy_bundle(),
        markets=[{"ticker": "KXWC26-XYZ", "title": "France vs Spain", "yes_ask": 5}],
        bankroll_cents=20000,
    )
    assert signals == []


def test_risk_blocks_signal_on_stop_loss() -> None:
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle=_toy_bundle(),
        markets=_markets(yes_ask=5),
        bankroll_cents=5000,  # down 75% from peak -> stop-loss
        peak_bankroll_cents=20000,
    )
    assert signals == []
