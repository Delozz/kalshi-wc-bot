"""Integration tests for signal generation (strategy/signal_gen.py) — all outcomes."""

from __future__ import annotations

import pandas as pd

from features import elo
from ingestion.api_football import Fixture
from strategy import signal_gen


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
        fixture_id=999,
        kickoff_utc="2026-06-20T18:00:00+00:00",
        status="NS",
        home_team="Brazil",
        away_team="Serbia",
        round="Group G",
    )


def _markets(home: int, draw: int, away: int) -> list[dict]:
    """Build synthetic market dicts using real API field names (cents args for readability)."""

    def _ask(cents: int) -> str:
        return f"{cents / 100:.4f}"

    return [
        {
            "ticker": "KXWC26-BRA-H",
            "title": "Brazil vs Serbia",
            "yes_sub_title": "Brazil",
            "yes_ask_dollars": _ask(home),
            "open_interest_fp": "10000.00",
        },
        {
            "ticker": "KXWC26-BRA-D",
            "title": "Brazil vs Serbia",
            "yes_sub_title": "Draw",
            "yes_ask_dollars": _ask(draw),
            "open_interest_fp": "10000.00",
        },
        {
            "ticker": "KXWC26-BRA-A",
            "title": "Brazil vs Serbia",
            "yes_sub_title": "Serbia",
            "yes_ask_dollars": _ask(away),
            "open_interest_fp": "10000.00",
        },
    ]


def _ratings(history: pd.DataFrame) -> dict[str, float]:
    return elo.final_ratings(history, use_tournament_k=True)


def _fixed_probs(_bundle, _features):
    return {"H": 0.6, "D": 0.3, "A": 0.2}


def test_resolver_maps_three_outcomes() -> None:
    resolved = signal_gen.default_outcome_resolver(_fixture(), _markets(50, 30, 20))
    assert resolved["H"] == ("KXWC26-BRA-H", 0.50)
    assert resolved["D"] == ("KXWC26-BRA-D", 0.30)
    assert resolved["A"] == ("KXWC26-BRA-A", 0.20)


def test_generates_a_signal_per_outcome_with_edge(monkeypatch) -> None:
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=_markets(5, 5, 5),  # cheap on every outcome -> edge on all three
        bankroll_cents=20000,
    )
    assert len(signals) == 3
    assert {s["market_ticker"] for s in signals} == {
        "KXWC26-BRA-H",
        "KXWC26-BRA-D",
        "KXWC26-BRA-A",
    }
    home = next(s for s in signals if s["market_ticker"] == "KXWC26-BRA-H")
    assert home["match_id"].split(":")[1] == "H"  # outcome encoded in match_id


def test_only_outcomes_with_edge_are_traded(monkeypatch) -> None:
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    # H: 0.60 vs 0.50 -> +0.10 (bet); D: 0.30 vs 0.40 -> -0.10 (skip); A: 0.20 vs 0.10 -> +0.10 (bet)
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=_markets(50, 40, 10),
        bankroll_cents=20000,
    )
    assert {s["market_ticker"] for s in signals} == {"KXWC26-BRA-H", "KXWC26-BRA-A"}


def test_no_signal_when_no_market_matches(monkeypatch) -> None:
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=[
            {"ticker": "X", "title": "France vs Spain", "yes_ask_dollars": "0.05"}
        ],
        bankroll_cents=20000,
    )
    assert signals == []


def test_scarce_slot_goes_to_highest_edge_not_first_processed(monkeypatch) -> None:
    # Only one position slot is free (n_open=2, cap=3). The away leg has the larger edge
    # (0.15) but is the LAST outcome processed; home is smaller (0.10) but processed first.
    # The pre-fix chronological code would have filled the slot with home; ranking by edge
    # first must instead award it to away.
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        # H: 0.60 vs 0.50 -> +0.10; D: 0.30 vs 0.40 -> -0.10 (skip); A: 0.20 vs 0.05 -> +0.15
        markets=_markets(50, 40, 5),
        bankroll_cents=20000,
        n_open=2,
    )
    assert len(signals) == 1
    assert signals[0]["market_ticker"] == "KXWC26-BRA-A"


def test_risk_blocks_on_stop_loss(monkeypatch) -> None:
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=_markets(5, 5, 5),
        bankroll_cents=5000,  # down 75% from peak -> stop-loss
        peak_bankroll_cents=20000,
    )
    assert signals == []
