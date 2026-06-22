"""Tests for the lineup-strength feature and its edge adjustment.

Pins the two correctness properties of the lineup-adjustment plan: the home-minus-away
sign convention / clamping in ``features.lineup`` and the zero-impact fallback plus
0–1 clamping in ``strategy.edge.apply_lineup_adjustment``.
"""

from __future__ import annotations

import math

from features import lineup
from strategy import edge


def _xi(*ratings: float | None) -> dict[str, object]:
    """Build a one-team lineup payload with the given startXI player ratings."""
    return {
        "team": {"name": "Testland"},
        "startXI": [{"player": {"id": i, "rating": r}} for i, r in enumerate(ratings)],
    }


def test_delta_zero_when_a_lineup_is_missing() -> None:
    assert lineup.lineup_strength_delta(None, _xi(7.0)) == 0.0
    assert lineup.lineup_strength_delta(_xi(7.0), None) == 0.0


def test_delta_zero_when_no_ratings_present() -> None:
    # An announced XI with no numeric ratings must not fabricate strength.
    assert lineup.lineup_strength_delta(_xi(None, None), _xi(7.0, 8.0)) == 0.0


def test_delta_is_signed_home_minus_away() -> None:
    delta = lineup.lineup_strength_delta(_xi(8.0, 7.0), _xi(6.0, 6.0))
    # home avg 7.5, away avg 6.0 -> (7.5 - 6.0) / 10
    assert math.isclose(delta, 0.15)
    # Reversing the sides flips the sign.
    assert math.isclose(
        lineup.lineup_strength_delta(_xi(6.0, 6.0), _xi(8.0, 7.0)), -0.15
    )


def test_delta_clamped_to_unit_range() -> None:
    assert lineup.lineup_strength_delta(_xi(50.0), _xi(0.0)) == 1.0
    assert lineup.lineup_strength_delta(_xi(0.0), _xi(50.0)) == -1.0


def test_adjustment_identity_at_zero_delta() -> None:
    # The zero-impact fallback: no announced lineup => probability is untouched.
    assert edge.apply_lineup_adjustment(0.42, 0.0) == 0.42


def test_adjustment_moves_probability_with_delta() -> None:
    weight = edge.settings.lineup_weight
    assert math.isclose(edge.apply_lineup_adjustment(0.50, 1.0), 0.50 * (1.0 + weight))
    assert edge.apply_lineup_adjustment(0.50, -1.0) < 0.50


def test_adjustment_clamped_to_probability_range() -> None:
    assert edge.apply_lineup_adjustment(0.99, 50.0) == 1.0
    assert edge.apply_lineup_adjustment(0.50, -50.0) == 0.0
