"""Tests for edge detection and signal assembly (strategy/edge.py)."""

from __future__ import annotations

import math

from strategy import edge


def test_compute_edge() -> None:
    assert math.isclose(edge.compute_edge(0.58, 0.51), 0.07)
    assert math.isclose(edge.compute_edge(0.40, 0.50), -0.10)


def test_has_edge_threshold() -> None:
    assert edge.has_edge(0.05, threshold=0.04) is True
    assert edge.has_edge(0.03, threshold=0.04) is False


def test_build_signal_none_below_threshold() -> None:
    signal = edge.build_signal(
        match_id="FRA_MAR_2026-06-20",
        market_ticker="KXWC26-FRA",
        model_prob=0.52,
        kalshi_yes_price=0.51,
        bankroll=1000.0,
        threshold=0.04,
    )
    assert signal is None


def test_build_signal_above_threshold() -> None:
    signal = edge.build_signal(
        match_id="FRA_MAR_2026-06-20",
        market_ticker="KXWC26-FRA",
        model_prob=0.62,
        kalshi_yes_price=0.50,
        bankroll=1000.0,
        threshold=0.04,
    )
    assert signal is not None
    assert math.isclose(signal["edge"], 0.12)
    assert signal["market_implied"] == 0.50
    assert signal["side"] == "YES"
    # Bet size is capped (5% of $1000 = 5000 cents) and positive.
    assert 0 < signal["bet_size_cents"] <= 5000
