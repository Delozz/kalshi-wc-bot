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
    decision = risk.check_all(
        bankroll=700.0,
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
