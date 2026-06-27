"""Tests for the live performance scorecard (dashboard/scorecard.py)."""

from __future__ import annotations

import math

from dashboard import scorecard


def _bet(
    *,
    leg: str = "A",
    team: str = "Japan",
    model: float = 0.40,
    market: float = 0.20,
    staked: int = 100,
    pnl: int,
) -> scorecard.BetResult:
    from features import confederation

    return scorecard.BetResult(
        match_id=f"1:{leg}:Brazil_{team}",
        leg=leg,
        team_backed=team,
        confed=confederation.confederation_of(team),
        model_prob=model,
        market_implied=market,
        edge=model - market,
        staked_cents=staked,
        pnl_cents=pnl,
        won=pnl > 0,
    )


def test_summary_roi_and_hit_rate() -> None:
    # Two losses (-100 each) and one win (+300 on a 100 stake): 1/3 hit, net +100 on 300.
    results = [
        _bet(pnl=-100),
        _bet(pnl=-100),
        _bet(pnl=300),
    ]
    s = scorecard.summarize(results)
    assert s.n_bets == 3
    assert s.n_wins == 1
    assert math.isclose(s.hit_rate, 1 / 3)
    assert s.staked_cents == 300
    assert s.pnl_cents == 100
    assert math.isclose(s.roi, 100 / 300)


def test_brier_model_vs_market() -> None:
    # Model says 40%, market says 20%, and the bet LOST (outcome 0). The market was closer to
    # reality, so its Brier must be the smaller of the two — the "no edge" signal.
    results = [_bet(model=0.40, market=0.20, pnl=-100)]
    s = scorecard.summarize(results)
    assert math.isclose(s.model_brier, 0.40**2)
    assert math.isclose(s.market_brier, 0.20**2)
    assert s.market_brier < s.model_brier


def test_empty_is_safe() -> None:
    s = scorecard.summarize([])
    assert s.n_bets == 0
    assert s.roi == 0.0
    assert s.model_brier == 0.0


def test_calibration_buckets_group_by_model_prob() -> None:
    # Two bets at ~40% (one win, one loss -> 50% realized) and one at ~20% (loss).
    results = [
        _bet(model=0.42, pnl=300),
        _bet(model=0.40, pnl=-100),
        _bet(model=0.20, pnl=-100),
    ]
    buckets = scorecard.calibration_buckets(results)
    by_band = {(b.low, b.high): b for b in buckets}
    mid = by_band[(0.4, 0.5)]
    assert mid.n == 2
    assert math.isclose(mid.win_rate, 0.5)
    low = by_band[(0.0, 0.3)]
    assert low.n == 1
    assert low.win_rate == 0.0


def test_breakdown_by_leg_and_confederation() -> None:
    results = [
        _bet(leg="A", team="Japan", pnl=-100),  # AFC
        _bet(leg="A", team="Iran", pnl=-100),  # AFC
        _bet(leg="H", team="Brazil", pnl=300),  # CONMEBOL (home leg backs Brazil)
    ]
    by_leg = scorecard.breakdown(results, "leg")
    assert by_leg["A"].n_bets == 2 and by_leg["A"].hit_rate == 0.0
    assert by_leg["H"].n_bets == 1 and by_leg["H"].hit_rate == 1.0

    by_confed = scorecard.breakdown(results, "confed")
    assert by_confed["AFC"].n_bets == 2
    assert by_confed["AFC"].roi < 0
    assert by_confed["CONMEBOL"].roi > 0


def test_decode_leg() -> None:
    assert scorecard._decode_leg("1:H:Brazil_Japan") == ("H", "Brazil")
    assert scorecard._decode_leg("1:A:Brazil_Japan") == ("A", "Japan")
    assert scorecard._decode_leg("1:D:Brazil_Japan") == ("D", None)
    assert scorecard._decode_leg("garbage") == ("?", None)
