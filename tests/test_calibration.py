"""Tests for the temperature calibration layer (model/calibration.py).

Pins the temper/fit contract behind the live DC path (L5): identity at tau=1, valid
renormalized distributions at any tau, and fit_temperature recovering the flattening
exponent for synthetically overconfident predictions.
"""

from __future__ import annotations

import math

import numpy as np

from model import calibration


def test_temper_probs_identity_at_one() -> None:
    probs = {"H": 0.5, "D": 0.3, "A": 0.2}
    assert calibration.temper_probs(probs, 1.0) == probs
    assert calibration.temper_probs({}, 0.5) == {}


def test_temper_probs_flattens_below_one_and_sharpens_above() -> None:
    probs = {"H": 0.6, "D": 0.25, "A": 0.15}
    flat = calibration.temper_probs(probs, 0.5)
    sharp = calibration.temper_probs(probs, 2.0)
    assert (
        flat["H"] < probs["H"] < sharp["H"]
    )  # top prob moves toward/away from uniform
    assert flat["A"] > probs["A"] > sharp["A"]
    assert math.isclose(sum(flat.values()), 1.0)
    assert math.isclose(sum(sharp.values()), 1.0)


def test_temper_matrix_matches_dict_path() -> None:
    matrix = np.array([[0.6, 0.25, 0.15], [0.2, 0.3, 0.5]])
    tempered = calibration.temper_matrix(matrix, 0.5)
    row0 = calibration.temper_probs({"H": 0.6, "D": 0.25, "A": 0.15}, 0.5)
    assert np.allclose(tempered[0], [row0["H"], row0["D"], row0["A"]])
    assert np.allclose(tempered.sum(axis=1), 1.0)


def test_fit_temperature_recovers_flattening_exponent() -> None:
    # Build overconfident predictions: true probabilities sharpened with tau=2. The fit
    # should recover ~0.5 (the inverse), flattening them back toward truth.
    rng = np.random.default_rng(7)
    n = 4000
    true = rng.dirichlet((4.0, 3.0, 2.0), size=n)
    labels = np.array([rng.choice(3, p=row) for row in true])
    overconfident = calibration.temper_matrix(true, 2.0)
    tau = calibration.fit_temperature(overconfident, labels)
    assert 0.35 < tau < 0.65


def test_fit_temperature_near_identity_for_calibrated_probs() -> None:
    # Predictions that ARE the true generating probabilities need no tempering.
    rng = np.random.default_rng(11)
    n = 4000
    true = rng.dirichlet((4.0, 3.0, 2.0), size=n)
    labels = np.array([rng.choice(3, p=row) for row in true])
    tau = calibration.fit_temperature(true, labels)
    assert 0.85 < tau < 1.15
