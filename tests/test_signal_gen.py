"""Integration tests for signal generation (strategy/signal_gen.py).

These exercise the one-bet-per-fixture policy: of a match's mutually-exclusive H/D/A legs,
only the single highest-edge admissible leg is bet (stacking correlated legs on one game is
the mistake that lost on DR Congo/France/Portugal). Most tests pin an explicit ``threshold``
so they stay decoupled from the operator-configured ``MIN_EDGE_THRESHOLD`` in ``.env``.
"""

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


def _fixture2() -> Fixture:
    return Fixture(
        fixture_id=888,
        kickoff_utc="2026-06-21T18:00:00+00:00",
        status="NS",
        home_team="France",
        away_team="Spain",
        round="Group H",
    )


def _ask(cents: int) -> str:
    return f"{cents / 100:.4f}"


def _named_markets(
    prefix: str, home_name: str, away_name: str, home: int, draw: int, away: int
) -> list[dict]:
    """Three per-outcome markets for one fixture, using real API field names (cents args)."""
    return [
        {
            "ticker": f"{prefix}-H",
            "title": f"{home_name} vs {away_name}",
            "yes_sub_title": home_name,
            "yes_ask_dollars": _ask(home),
            "open_interest_fp": "10000.00",
        },
        {
            "ticker": f"{prefix}-D",
            "title": f"{home_name} vs {away_name}",
            "yes_sub_title": "Draw",
            "yes_ask_dollars": _ask(draw),
            "open_interest_fp": "10000.00",
        },
        {
            "ticker": f"{prefix}-A",
            "title": f"{home_name} vs {away_name}",
            "yes_sub_title": away_name,
            "yes_ask_dollars": _ask(away),
            "open_interest_fp": "10000.00",
        },
    ]


def _markets(home: int, draw: int, away: int) -> list[dict]:
    return _named_markets("KXWC26-BRA", "Brazil", "Serbia", home, draw, away)


def _ratings(history: pd.DataFrame) -> dict[str, float]:
    return elo.final_ratings(history, use_tournament_k=True)


def _fixed_probs(_bundle, _features):
    # A normalized H/D/A distribution (real engines always sum to 1; the confederation and
    # squad priors renormalize, so an un-normalized fixture would shift under them).
    return {"H": 0.55, "D": 0.27, "A": 0.18}


def _leg(signal) -> str:
    return signal["match_id"].split(":")[1]


def test_resolver_maps_three_outcomes() -> None:
    resolved = signal_gen.default_outcome_resolver(_fixture(), _markets(50, 30, 20))
    assert resolved["H"] == ("KXWC26-BRA-H", 0.50)
    assert resolved["D"] == ("KXWC26-BRA-D", 0.30)
    assert resolved["A"] == ("KXWC26-BRA-A", 0.20)


def test_one_bet_per_fixture_takes_highest_edge(monkeypatch) -> None:
    # All three legs clear the edge threshold and the ratio guards, but only ONE bet is
    # placed per match — the highest-edge leg. H: 0.55 vs 0.40 -> +0.15 wins the slot over
    # D (+0.09) and A (+0.06).
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=_markets(40, 18, 12),
        bankroll_cents=20000,
        threshold=0.04,
    )
    assert len(signals) == 1
    assert _leg(signals[0]) == "H"
    assert signals[0]["market_ticker"] == "KXWC26-BRA-H"


def test_only_outcome_with_edge_is_traded(monkeypatch) -> None:
    # H has no edge (0.55 vs 0.60) and A is below threshold (0.18 vs 0.16 -> +0.02), so the
    # draw is the only qualifying leg and therefore the bet — a non-favorite leg can win the
    # single slot when the favorite offers no value.
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=_markets(60, 20, 16),
        bankroll_cents=20000,
        threshold=0.04,
    )
    assert [_leg(s) for s in signals] == ["D"]


def test_overpriced_underdog_leg_is_filtered(monkeypatch) -> None:
    # A (0.18 vs 0.10) has the largest raw edge (+0.08) but a 1.8x model/market ratio, so the
    # tightened underdog guard drops it before ranking. The favorite H (+0.05) is then the
    # only admissible leg — proving A was excluded by the ratio, not merely out-ranked.
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=_markets(50, 40, 10),
        bankroll_cents=20000,
        threshold=0.04,
    )
    assert [_leg(s) for s in signals] == ["H"]


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
        threshold=0.04,
    )
    assert signals == []


def test_scarce_slot_goes_to_highest_edge_fixture(monkeypatch) -> None:
    # Only one position slot is free (n_open=4, cap=5). Two fixtures each offer a favorite
    # leg, but France's edge (+0.20) beats Brazil's (+0.15). The single slot must go to the
    # higher-edge fixture's bet, regardless of fixture processing order.
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    markets = _markets(45, 20, 13) + _named_markets(
        "KXWC26-FRA", "France", "Spain", 40, 20, 13
    )
    signals = signal_gen.generate_signals(
        fixtures=[_fixture(), _fixture2()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=markets,
        bankroll_cents=20000,
        n_open=4,
        threshold=0.04,
    )
    assert len(signals) == 1
    assert signals[0]["market_ticker"] == "KXWC26-FRA-H"


def test_injected_predictor_overrides_classifier() -> None:
    # The engine seam: a custom predictor supplies the base probs instead of the bundle, and
    # the rest of the pipeline (edge, sizing, one-per-fixture cap) runs on them. This is the
    # path run_live uses to swap in the Dixon-Coles engine.
    history = _history()
    used = {}

    def predictor(_fixture, _features):
        used["called"] = True
        return {"H": 0.55, "D": 0.27, "A": 0.18}

    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        predictor=predictor,
        markets=_markets(30, 20, 13),
        bankroll_cents=20000,
        threshold=0.04,
    )
    assert used.get("called") is True
    assert [_leg(s) for s in signals] == ["H"]


def test_dc_predictor_drives_signals() -> None:
    # A fitted Dixon-Coles model plugged in as the predictor produces a signal end to end.
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
        threshold=0.04,
    )
    # Brazil is the stronger side in the fitted model, so the single bet must be the home leg.
    assert [_leg(s) for s in signals] == ["H"]


def test_powerhouse_blocks_draw_and_upset_legs(monkeypatch) -> None:
    # France (2000 ELO) vs Iraq (1600 ELO): a 400-point gap. The draw and Iraq-upset legs are
    # filtered as powerhouse mismatches, so only the favorite (France) leg can survive — and
    # under the one-per-fixture cap that single France leg is the bet. The Iraq-over-France
    # bet we never want again.
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
    markets = _named_markets("KXWC26-FRA", "France", "Iraq", 50, 8, 8)
    signals = signal_gen.generate_signals(
        fixtures=[fixture],
        history=_history(),
        ratings={"France": 2000.0, "Iraq": 1600.0},
        bundle={},
        markets=markets,
        bankroll_cents=20000,
        threshold=0.04,
    )
    assert [_leg(s) for s in signals] == ["H"]


def test_squad_prior_flips_bet_from_draw_to_favorite(monkeypatch) -> None:
    # Without a squad prior the draw is the highest-edge admissible leg and gets the single
    # slot. Feeding a prior where the home side (Brazil) is much stronger tilts probability
    # onto the favourite: the draw's edge collapses and the bet flips to the home leg. The
    # bot never bets the draw against a clearly stronger squad. (The tilt mechanic itself is
    # unit-tested in test_squad.py.)
    monkeypatch.setattr(
        signal_gen.predict_mod,
        "predict_outcome",
        lambda _b, _f: {"H": 0.45, "D": 0.35, "A": 0.20},
    )
    history = _history()
    ratings = _ratings(history)
    markets = _markets(34, 23, 15)

    base = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=ratings,
        bundle={},
        markets=markets,
        bankroll_cents=20000,
        threshold=0.04,
    )
    assert [_leg(s) for s in base] == ["D"]

    tilted = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=ratings,
        bundle={},
        markets=markets,
        bankroll_cents=20000,
        threshold=0.04,
        squad_ratings_by_team={
            "Brazil": {1: 8.0, 2: 8.0},
            "Serbia": {1: 6.0, 2: 6.0},
        },
    )
    assert [_leg(s) for s in tilted] == ["H"]


def test_held_fixture_blocks_all_its_legs(monkeypatch) -> None:
    # One bet per fixture across cycles: holding any leg of a match (here the H leg) bars its
    # sibling D/A legs too, so the bot never accumulates two correlated legs on one game even
    # when a persistent edge re-appears next cycle. The other legs would otherwise qualify.
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=_markets(40, 20, 13),
        bankroll_cents=20000,
        held_tickers={"KXWC26-BRA-H"},
        threshold=0.04,
    )
    assert signals == []


def test_confederation_correction_suppresses_cross_confed_upset(monkeypatch) -> None:
    # Brazil (CONMEBOL) vs Japan (AFC). The raw model gives Japan 0.318, which at the 0.24
    # line would clear the edge (+0.078) and be bet — the exact Japan-over-Brazil mistake.
    # The confederation tilt (+143 ELO for Brazil) pulls Japan to ~0.19, killing that edge,
    # and lifts Brazil's home leg so the single bet flips to the favorite instead.
    monkeypatch.setattr(
        signal_gen.predict_mod,
        "predict_outcome",
        lambda _b, _f: {"H": 0.385, "D": 0.298, "A": 0.318},
    )
    fixture = Fixture(
        fixture_id=7,
        kickoff_utc="2026-06-29T18:00:00+00:00",
        status="NS",
        home_team="Brazil",
        away_team="Japan",
        round="Group C",
    )
    markets = _named_markets("KXWC26-BRJ", "Brazil", "Japan", 45, 30, 24)
    signals = signal_gen.generate_signals(
        fixtures=[fixture],
        history=_history(),
        ratings={"Brazil": 1940.0, "Japan": 1884.0},
        bundle={},
        markets=markets,
        bankroll_cents=20000,
        threshold=0.04,
    )
    legs = {_leg(s) for s in signals}
    assert "A" not in legs  # Japan upset suppressed by the confederation correction
    assert legs == {"H"}  # the favorite (Brazil) is the single bet


def test_risk_blocks_on_stop_loss(monkeypatch) -> None:
    # Candidates clear the edge/ratio guards, but a 75% drawdown from peak trips the
    # stop-loss in the ranking phase and nothing is bet.
    monkeypatch.setattr(signal_gen.predict_mod, "predict_outcome", _fixed_probs)
    history = _history()
    signals = signal_gen.generate_signals(
        fixtures=[_fixture()],
        history=history,
        ratings=_ratings(history),
        bundle={},
        markets=_markets(40, 20, 13),
        bankroll_cents=5000,  # down 75% from peak -> stop-loss
        peak_bankroll_cents=20000,
        threshold=0.04,
    )
    assert signals == []
