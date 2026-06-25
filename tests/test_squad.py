"""Tests for the squad-strength prior (features/squad.py + strategy/edge.apply_squad_prior)."""

from __future__ import annotations

from features import squad
from strategy import edge


def test_squad_strength_uses_top_n() -> None:
    # Top-2 mean of {9, 8, 1, 1} is 8.5; the weak tail is ignored.
    ratings = {1: 9.0, 2: 8.0, 3: 1.0, 4: 1.0}
    assert squad.squad_strength(ratings, top_n=2) == 8.5


def test_squad_strength_none_when_unrated() -> None:
    assert squad.squad_strength(None) is None
    assert squad.squad_strength({}) is None


def test_delta_sign_and_scale() -> None:
    # Home mean 8.0 vs away mean 6.0 -> (8-6)/10 = +0.20, home is stronger.
    home = {1: 8.0, 2: 8.0}
    away = {1: 6.0, 2: 6.0}
    assert squad.squad_strength_delta(home, away, top_n=2) == 0.20
    assert squad.squad_strength_delta(away, home, top_n=2) == -0.20


def test_delta_zero_when_either_side_unrated() -> None:
    assert squad.squad_strength_delta({1: 7.0}, None) == 0.0
    assert squad.squad_strength_delta(None, {1: 7.0}) == 0.0


def test_prior_is_identity_when_delta_zero() -> None:
    probs = {"H": 0.5, "D": 0.3, "A": 0.2}
    assert edge.apply_squad_prior(probs, 0.0, weight=3.0) == probs


def test_prior_tilts_toward_stronger_home_squad() -> None:
    # Positive delta favours home: H rises, BOTH D and A fall, vector still sums to 1.
    probs = {"H": 0.40, "D": 0.35, "A": 0.25}
    out = edge.apply_squad_prior(probs, 0.10, weight=3.0)
    assert out["H"] > probs["H"]
    assert out["D"] < probs["D"]
    assert out["A"] < probs["A"]
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_prior_tilts_toward_stronger_away_squad() -> None:
    # Negative delta favours away: A rises, H and D fall.
    probs = {"H": 0.40, "D": 0.35, "A": 0.25}
    out = edge.apply_squad_prior(probs, -0.10, weight=3.0)
    assert out["A"] > probs["A"]
    assert out["H"] < probs["H"]
    assert out["D"] < probs["D"]
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_prior_keeps_probabilities_valid_at_extreme() -> None:
    # A huge tilt must never produce a negative or >1 probability.
    probs = {"H": 0.40, "D": 0.35, "A": 0.25}
    out = edge.apply_squad_prior(probs, 1.0, weight=10.0)
    assert all(0.0 <= p <= 1.0 for p in out.values())
    assert abs(sum(out.values()) - 1.0) < 1e-9
