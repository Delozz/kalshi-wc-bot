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
        # Cheap on every outcome -> edge on all three, but kept above the 6c floor and
        # within the 2.5x model/market band so the signal-quality guards are no-ops here.
        markets=_markets(30, 15, 10),
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
    # Only one position slot is free (n_open=4, cap=5). The away leg has the larger edge
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
        # H: 0.60 vs 0.50 -> +0.10; D: 0.30 vs 0.40 -> -0.10 (skip); A: 0.20 vs 0.09 -> +0.11.
        # Away leg stays above the 6c floor (9c) so it remains the highest-edge candidate.
        markets=_markets(50, 40, 9),
        bankroll_cents=20000,
        n_open=4,
    )
    assert len(signals) == 1
    assert signals[0]["market_ticker"] == "KXWC26-BRA-A"


def test_injected_predictor_overrides_classifier() -> None:
    # The engine seam: a custom predictor supplies the base probs instead of the bundle,
    # and the rest of the pipeline (edge, sizing) runs on them unchanged. This is the path
    # run_live uses to swap in the Dixon-Coles engine.
    history = _history()
    used = {}

    def predictor(_fixture, _features):
        used["called"] = True
        return {"H": 0.6, "D": 0.3, "A": 0.2}

    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        predictor=predictor,
        markets=_markets(30, 15, 10),
        bankroll_cents=20000,
    )
    assert used.get("called") is True
    assert {_leg(s) for s in signals} == {"H", "D", "A"}


def test_dc_predictor_drives_signals() -> None:
    # A fitted Dixon-Coles model plugged in as the predictor produces signals end to end.
    import pandas as pd

    from model import dixon_coles

    scorelines = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
            "home_team": ["Brazil", "Serbia", "Brazil", "Serbia"],
            "away_team": ["Serbia", "Brazil", "Serbia", "Brazil"],
            "fthg": [3, 0, 2, 1],
            "ftag": [0, 2, 1, 3],
            "neutral": [True, True, True, True],
        }
    )
    model = dixon_coles.fit(scorelines, min_matches=2)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        predictor=signal_gen._dc_predictor(model),
        markets=_markets(30, 25, 25),
        bankroll_cents=20000,
    )
    # Brazil is the stronger side in the fitted model, so the home leg must be among signals.
    assert "H" in {_leg(s) for s in signals}


def test_powerhouse_blocks_draw_and_upset_legs(monkeypatch) -> None:
    # France (2000 ELO) vs Iraq (1600 ELO): a 400-point gap. Even though the model gives
    # the draw and the Iraq upset enough probability to clear the edge threshold at these
    # prices, both legs must be filtered as powerhouse mismatches — only the favorite
    # (France) leg may survive. This is the Iraq-over-France bet we never want again.
    monkeypatch.setattr(
        signal_gen.predict_mod,
        "predict_outcome",
        lambda _b, _f: {"H": 0.65, "D": 0.20, "A": 0.15},
    )
    fixture = Fixture(
        fixture_id=1,
        kickoff_utc="2026-06-22T18:00:00+00:00",
        status="NS",
        home_team="France",
        away_team="Iraq",
        round="Group A",
    )
    markets = [
        {
            "ticker": "KXWC26-FRA-H",
            "title": "France vs Iraq",
            "yes_sub_title": "France",
            "yes_ask_dollars": "0.5000",
            "open_interest_fp": "10000.00",
        },
        {
            "ticker": "KXWC26-FRA-D",
            "title": "France vs Iraq",
            "yes_sub_title": "Draw",
            "yes_ask_dollars": "0.0800",
            "open_interest_fp": "10000.00",
        },
        {
            "ticker": "KXWC26-FRA-A",
            "title": "France vs Iraq",
            "yes_sub_title": "Iraq",
            "yes_ask_dollars": "0.0800",
            "open_interest_fp": "10000.00",
        },
    ]
    signals = signal_gen.generate_signals(
        fixtures=[fixture],
        history=_history(),
        ratings={"France": 2000.0, "Iraq": 1600.0},
        bundle={},
        markets=markets,
        bankroll_cents=20000,
    )
    assert {s["match_id"].split(":")[1] for s in signals} == {"H"}


def _leg(signal) -> str:
    return signal["match_id"].split(":")[1]


def test_squad_prior_suppresses_bets_against_stronger_squad(monkeypatch) -> None:
    # Without a squad prior, all three legs clear the edge at these prices. Feeding a squad
    # prior where the home side (Brazil) is much stronger than the away side (Serbia) must
    # tilt probability onto the favourite: the draw and the upset legs lose enough mass to
    # fall below the market and stop being bet, while the favourite leg survives. This is
    # the Portugal/Norway fix — we no longer bet against a clearly stronger squad.
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    ratings = _ratings(history)
    markets = _markets(40, 20, 15)

    base = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=ratings,
        bundle={},
        markets=markets,
        bankroll_cents=20000,
    )
    assert {_leg(s) for s in base} == {"H", "D", "A"}

    tilted = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=ratings,
        bundle={},
        markets=markets,
        bankroll_cents=20000,
        squad_ratings_by_team={
            "Brazil": {1: 8.0, 2: 8.0},
            "Serbia": {1: 6.0, 2: 6.0},
        },
    )
    legs = {_leg(s) for s in tilted}
    assert "H" in legs
    assert "D" not in legs and "A" not in legs


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
