"""Tests for the lineup-strength feature and its edge adjustment.

Pins the two correctness properties of the lineup-adjustment plan: the home-minus-away
sign convention / clamping in ``features.lineup`` and the zero-impact fallback plus
vector renormalization in ``strategy.edge.apply_lineup_prior``.
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


def test_lineup_prior_identity_at_zero_delta() -> None:
    # The zero-impact fallback: no announced lineup => the vector is untouched.
    probs = {"H": 0.42, "D": 0.30, "A": 0.28}
    assert edge.apply_lineup_prior(probs, 0.0) == probs


def test_lineup_prior_tilts_and_renormalizes() -> None:
    # A stronger home XI lifts H, shrinks D and A, and the vector still sums to 1 —
    # the old per-leg nudge could push the three legs above a total of 1.
    probs = {"H": 0.40, "D": 0.30, "A": 0.30}
    tilted = edge.apply_lineup_prior(probs, 0.5, weight=0.10)
    assert tilted["H"] > probs["H"]
    assert tilted["D"] < probs["D"]
    assert tilted["A"] < probs["A"]
    assert math.isclose(sum(tilted.values()), 1.0)
    # Reversing the sides tilts the other way.
    reversed_tilt = edge.apply_lineup_prior(probs, -0.5, weight=0.10)
    assert reversed_tilt["A"] > probs["A"]


def test_lineup_prior_extreme_delta_stays_valid() -> None:
    # Even an absurd delta produces a valid distribution (shrink factor is capped).
    probs = {"H": 0.40, "D": 0.30, "A": 0.30}
    tilted = edge.apply_lineup_prior(probs, 50.0, weight=1.0)
    assert all(0.0 < p < 1.0 for p in tilted.values())
    assert math.isclose(sum(tilted.values()), 1.0)
