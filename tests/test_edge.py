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


def test_blend_with_book_shrinks_toward_anchor() -> None:
    probs = {"H": 0.50, "D": 0.30, "A": 0.20}
    anchor = {"H": 0.40, "D": 0.35, "A": 0.25}
    blended = edge.blend_with_book(probs, anchor, weight=0.3)
    # 0.3*model + 0.7*anchor, per leg; both inputs sum to 1 so no renorm distortion.
    assert math.isclose(blended["H"], 0.43)
    assert math.isclose(blended["D"], 0.335)
    assert math.isclose(blended["A"], 0.235)
    assert math.isclose(sum(blended.values()), 1.0)


def test_blend_with_book_identity_fallbacks() -> None:
    probs = {"H": 0.5, "D": 0.3, "A": 0.2}
    # No anchor, incomplete anchor, or full model weight -> unchanged (zero-impact).
    assert edge.blend_with_book(probs, None, weight=0.3) == probs
    assert edge.blend_with_book(probs, {"H": 0.4, "D": 0.35}, weight=0.3) == probs
    assert (
        edge.blend_with_book(probs, {"H": 0.4, "D": 0.35, "A": 0.25}, weight=1.0)
        == probs
    )


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
