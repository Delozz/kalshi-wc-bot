"""Tests for Kelly sizing (strategy/kelly.py) — half-Kelly and the hard cap (L6)."""

from __future__ import annotations

import math

from strategy import kelly


def test_zero_fraction_with_no_edge() -> None:
    # Model agrees with the price -> no edge -> no bet.
    assert kelly.kelly_fraction(0.50, 0.50) == 0.0
    # Model below the price -> negative edge -> clamped to 0.
    assert kelly.kelly_fraction(0.40, 0.50) == 0.0


def test_positive_fraction_with_edge() -> None:
    assert kelly.kelly_fraction(0.60, 0.50) > 0.0


def test_kelly_formula_value() -> None:
    # price=0.50 -> b=1; f* = p - (1-p)/b = 0.60 - 0.40 = 0.20.
    assert math.isclose(kelly.kelly_fraction(0.60, 0.50), 0.20)


def test_invalid_price_returns_zero() -> None:
    assert kelly.kelly_fraction(0.6, 0.0) == 0.0
    assert kelly.kelly_fraction(0.6, 1.0) == 0.0
    assert kelly.kelly_fraction(0.6, 1.5) == 0.0


def test_half_kelly_applied() -> None:
    # Full kelly 0.20, half-kelly 0.10 — exceeds the 5% cap, so the cap dominates.
    sizing = kelly.half_kelly_size(
        0.60, 0.50, 1000.0, fraction=0.5, max_bet_fraction=0.05
    )
    assert math.isclose(sizing.full_kelly, 0.20)
    assert math.isclose(sizing.used_fraction, 0.05)


def test_hard_cap_never_exceeded() -> None:
    # Enormous edge -> full kelly large -> half-kelly large -> still capped at 5%.
    sizing = kelly.half_kelly_size(
        0.99, 0.10, 1000.0, fraction=0.5, max_bet_fraction=0.05
    )
    assert sizing.used_fraction <= 0.05
    assert sizing.bet_size <= 0.05 * 1000.0


def test_half_kelly_below_cap_uses_half() -> None:
    # Small edge so half-kelly stays under the cap and is used as-is.
    full = kelly.kelly_fraction(0.52, 0.50)
    sizing = kelly.half_kelly_size(
        0.52, 0.50, 1000.0, fraction=0.5, max_bet_fraction=0.05
    )
    assert math.isclose(sizing.used_fraction, full * 0.5)
