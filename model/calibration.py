"""Probability calibration (model/calibration.py).

L5 — Calibration Before Kelly: a model that says 70% must be right ~70% of the time.
Uncalibrated probabilities destroy Kelly sizing, so calibration is applied to every
model before any sizing code runs. Platt scaling (``method="sigmoid"``) by default.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

_EPS = 1e-12  # probability floor so log()/power never sees zero


def temper_probs(probs: Mapping[str, float], tau: float) -> dict[str, float]:
    """Temperature-scale a probability dict: ``p_i ** tau``, renormalized.

    ``tau < 1`` flattens an overconfident vector toward uniform; ``tau > 1`` sharpens;
    ``tau == 1`` is the identity (the zero-impact fallback). This is the single-parameter
    calibration layer the live Dixon-Coles path applies (L5) — one parameter because the
    fit sample (one World Cup, ~64 matches) cannot support a richer map without overfitting.
    """
    if tau == 1.0 or not probs:
        return dict(probs)
    powered = {k: max(float(v), _EPS) ** tau for k, v in probs.items()}
    total = sum(powered.values())
    if total <= 0.0:
        return dict(probs)
    return {k: v / total for k, v in powered.items()}


def temper_matrix(probs: np.ndarray, tau: float) -> np.ndarray:
    """Row-wise :func:`temper_probs` for an ``(n, k)`` probability matrix."""
    powered = np.clip(probs, _EPS, None) ** tau
    return powered / powered.sum(axis=1, keepdims=True)


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.25, 2.0),
) -> float:
    """The exponent ``tau`` minimizing multiclass log-loss of the tempered ``probs``.

    Fit ONLY on out-of-sample predictions (the 2018 dev tournament) — fitting on training
    rows would just certify the model's in-sample confidence. ``labels`` are integer class
    indices into ``probs``' columns.
    """
    labels = np.asarray(labels, dtype=int)

    def nll(tau: float) -> float:
        tempered = temper_matrix(probs, tau)
        picked = tempered[np.arange(len(labels)), labels]
        return float(-np.mean(np.log(np.clip(picked, _EPS, None))))

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x)


def calibrate(
    estimator: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Fit a cross-validated calibrated classifier around ``estimator``.

    The estimator is cloned and refit within each CV fold, and the calibration map is
    learned on the held-out folds — so the returned classifier is calibrated without
    leaking the training labels into the calibration map.
    """
    calibrated = CalibratedClassifierCV(estimator, method=method, cv=cv)
    calibrated.fit(x_train, y_train)
    return calibrated
