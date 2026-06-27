"""Tests for the confederation-drift correction.

Covers features/confederation.py (the offset map) and strategy/edge.apply_confederation_prior
(the ELO-odds-ratio tilt applied to the H/D/A vector).
"""

from __future__ import annotations

import math

from features import confederation as cf
from strategy import edge


def test_offset_for_known_and_unknown() -> None:
    assert cf.offset_for("Brazil") == 64.0  # CONMEBOL
    assert cf.offset_for("Japan") == -79.0  # AFC
    assert cf.offset_for("France") == 62.0  # UEFA
    # An unmapped team is a zero-impact fallback, never a fabricated adjustment.
    assert cf.offset_for("Atlantis") == 0.0


def test_elo_delta_signs_and_intra_confederation_cancels() -> None:
    # Cross-confederation: the full differential, positive favouring the home side.
    assert cf.elo_delta("Brazil", "Japan") == 64.0 - (-79.0)
    assert cf.elo_delta("Japan", "Brazil") == -143.0
    # Same confederation -> exactly 0, so an all-AFC or all-UEFA game is never adjusted.
    assert cf.elo_delta("Brazil", "Argentina") == 0.0
    assert cf.elo_delta("France", "Spain") == 0.0


def test_apply_prior_identity_cases() -> None:
    probs = {"H": 0.385, "D": 0.298, "A": 0.318}
    # Zero delta (same confederation / unmapped) and zero weight are both the identity.
    assert edge.apply_confederation_prior(probs, 0.0) == probs
    assert edge.apply_confederation_prior(probs, 143.0, weight=0.0) == probs
    assert edge.apply_confederation_prior({}, 143.0) == {}


def test_apply_prior_tilts_toward_stronger_confederation() -> None:
    # Brazil (CONMEBOL) home vs Japan (AFC) away, +143 ELO delta: the home win prob rises,
    # the away (Japan) win prob falls toward the market, and the vector still sums to 1.
    probs = {"H": 0.385, "D": 0.298, "A": 0.318}
    out = edge.apply_confederation_prior(probs, 143.0, weight=1.0)
    assert math.isclose(sum(out.values()), 1.0)
    assert out["H"] > probs["H"]
    assert out["A"] < probs["A"]
    # The home/away odds ratio is multiplied by exactly 10 ** (delta / 400).
    factor = 10.0 ** (143.0 / 400.0)
    assert math.isclose(out["H"] / out["A"], (probs["H"] / probs["A"]) * factor)
    # Lands at the market's Japan price (~0.19) — the empirical target.
    assert math.isclose(out["A"], 0.193, abs_tol=0.005)


def test_apply_prior_is_symmetric_in_delta_sign() -> None:
    # Flipping the delta sign mirrors the tilt onto the other side (H/A-symmetric input).
    probs = {"H": 0.35, "D": 0.30, "A": 0.35}
    pos = edge.apply_confederation_prior(probs, 100.0, weight=1.0)
    neg = edge.apply_confederation_prior(probs, -100.0, weight=1.0)
    assert math.isclose(pos["H"], neg["A"])
    assert math.isclose(pos["A"], neg["H"])
    assert math.isclose(pos["D"], neg["D"])
