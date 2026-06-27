"""Tests for risk controls (strategy/risk.py)."""

from __future__ import annotations

from strategy import risk


def test_stop_loss_triggers_at_threshold() -> None:
    # 25% drop from a peak of 1000 -> 750 triggers.
    assert risk.stop_loss_triggered(750.0, 1000.0, threshold=0.25) is True
    assert risk.stop_loss_triggered(800.0, 1000.0, threshold=0.25) is False


def test_exposure_cap() -> None:
    # 20% of 1000 = 200 cap. 150 open + 40 new = 190 ok; +60 = 210 not ok.
    assert risk.exposure_ok(150.0, 40.0, 1000.0, max_exposure=0.20) is True
    assert risk.exposure_ok(150.0, 60.0, 1000.0, max_exposure=0.20) is False


def test_position_count() -> None:
    assert risk.position_count_ok(2, max_positions=3) is True
    assert risk.position_count_ok(3, max_positions=3) is False


def test_liquidity_floor() -> None:
    assert risk.liquidity_ok(5000.0) is True
    assert risk.liquidity_ok(4999.0) is False


def test_price_stable() -> None:
    assert risk.price_stable(0.50, 0.52, max_move=0.03) is True
    assert risk.price_stable(0.50, 0.55, max_move=0.03) is False


def test_check_all_priority_stop_loss_first() -> None:
    # Deep 90% drawdown so stop-loss trips regardless of the operator-configured
    # threshold (this asserts stop-loss *priority* in check_all, not a specific level).
    decision = risk.check_all(
        bankroll=100.0,
        peak_bankroll=1000.0,
        open_exposure=0.0,
        new_bet=10.0,
        n_open=0,
        open_interest=10000.0,
    )
    assert decision.approved is False
    assert decision.reason == "stop_loss"


def test_check_all_approves_clean_trade() -> None:
    decision = risk.check_all(
        bankroll=1000.0,
        peak_bankroll=1000.0,
        open_exposure=50.0,
        new_bet=40.0,
        n_open=1,
        open_interest=10000.0,
    )
    assert decision.approved is True
    assert decision.reason == "ok"


def test_price_floor() -> None:
    assert risk.price_floor_ok(0.06, min_price=0.06) is True
    assert risk.price_floor_ok(0.05, min_price=0.06) is False


def test_mismatch_ratio() -> None:
    # model_prob / price: 0.20/0.10 = 2.0 ok; 0.30/0.10 = 3.0 too high.
    assert risk.mismatch_ok(0.20, 0.10, max_ratio=2.5) is True
    assert risk.mismatch_ok(0.30, 0.10, max_ratio=2.5) is False
    assert risk.mismatch_ok(0.20, 0.0) is False  # zero price is never tradeable


def test_underdog_ratio_is_tighter_than_favorite() -> None:
    # A model that prices a leg at 0.40 against a 0.22 line is 1.82x the market: blocked as
    # an overpriced draw/underdog leg (the Iran/Paraguay/Japan losses), but allowed on the
    # favorite leg, which uses the looser 2.5x bar and isn't where the model overprices.
    under = risk.outcome_admissible(
        bet_on_favorite=False,
        model_prob=0.40,
        market_price=0.22,
        favorite_elo_gap=50.0,
    )
    assert under.approved is False and under.reason == "model_market_mismatch"
    fav = risk.outcome_admissible(
        bet_on_favorite=True,
        model_prob=0.40,
        market_price=0.22,
        favorite_elo_gap=50.0,
    )
    assert fav.approved is True


def test_favorite_not_overwhelming() -> None:
    # The favorite's own leg always passes, however large the gap.
    assert risk.favorite_not_overwhelming(True, 500.0, max_gap=200.0) is True
    # A draw/upset leg is blocked once the gap reaches the threshold, allowed below it.
    assert risk.favorite_not_overwhelming(False, 250.0, max_gap=200.0) is False
    assert risk.favorite_not_overwhelming(False, 150.0, max_gap=200.0) is True


def test_outcome_admissible_reasons() -> None:
    # Clean underdog leg in a close matchup: passes all three guards (1.33x < 1.6x).
    ok = risk.outcome_admissible(
        bet_on_favorite=False, model_prob=0.16, market_price=0.12, favorite_elo_gap=50.0
    )
    assert ok.approved is True and ok.reason == "ok"
    # Below the 6c price floor (the Iraq-win-at-3c longshot).
    floor = risk.outcome_admissible(
        bet_on_favorite=False, model_prob=0.08, market_price=0.03, favorite_elo_gap=50.0
    )
    assert floor.approved is False and floor.reason == "below_price_floor"
    # Model dwarfs the line (>1.6x underdog bar) even above the floor.
    mismatch = risk.outcome_admissible(
        bet_on_favorite=False, model_prob=0.30, market_price=0.10, favorite_elo_gap=50.0
    )
    assert mismatch.approved is False and mismatch.reason == "model_market_mismatch"
    # Draw/upset vs an overwhelming favorite. Price 0.12 keeps the leg under the 1.6x ratio
    # bar so the powerhouse-gap guard (not the ratio) is the one that rejects it.
    power = risk.outcome_admissible(
        bet_on_favorite=False,
        model_prob=0.17,
        market_price=0.12,
        favorite_elo_gap=303.0,
    )
    assert power.approved is False and power.reason == "powerhouse_favorite"


def test_outcome_admissible_combined_elo_squad_filter() -> None:
    # ELO gap 182 is under the pure-ELO threshold (200), so on its own it passes (price 0.16
    # keeps the 0.25 model leg under the 1.6x ratio bar so the ratio guard isn't the gate)...
    elo_only = risk.outcome_admissible(
        bet_on_favorite=False,
        model_prob=0.25,
        market_price=0.16,
        favorite_elo_gap=182.0,
    )
    assert elo_only.approved is True
    # ...but once the squad prior confirms the same favorite, the lower bar (150) blocks it
    # (the Portugal-vs-Uzbekistan tie: ELO 182 + a stronger Portugal squad).
    with_squad = risk.outcome_admissible(
        bet_on_favorite=False,
        model_prob=0.25,
        market_price=0.16,
        favorite_elo_gap=182.0,
        squad_confirms_favorite=True,
    )
    assert with_squad.approved is False
    assert with_squad.reason == "powerhouse_favorite_squad"
    # Below the squad-confirmed bar (150), a confirming squad still doesn't suppress.
    small_gap = risk.outcome_admissible(
        bet_on_favorite=False,
        model_prob=0.25,
        market_price=0.16,
        favorite_elo_gap=120.0,
        squad_confirms_favorite=True,
    )
    assert small_gap.approved is True
    # The favorite's own leg is never suppressed, even with a confirming squad.
    fav_leg = risk.outcome_admissible(
        bet_on_favorite=True,
        model_prob=0.65,
        market_price=0.50,
        favorite_elo_gap=182.0,
        squad_confirms_favorite=True,
    )
    assert fav_leg.approved is True
