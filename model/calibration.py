"""Probability calibration (model/calibration.py).

L5 — Calibration Before Kelly: a model that says 70% must be right ~70% of the time.
Uncalibrated probabilities destroy Kelly sizing, so calibration is applied to every
model before any sizing code runs. Platt scaling (``method="sigmoid"``) by default.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV


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
