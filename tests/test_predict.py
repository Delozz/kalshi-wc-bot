"""Tests for model inference (model/predict.py) using a small in-test model."""

from __future__ import annotations

import math

import numpy as np

from model import dataset, predict
from model.baseline import build_baseline


def _toy_bundle() -> dict:
    cols = dataset.FEATURE_COLUMNS
    rng = np.random.RandomState(0)
    x = rng.randn(90, len(cols))
    y = np.array([0, 1, 2] * 30)
    estimator = build_baseline().fit(x, y)
    return {
        "feature_columns": cols,
        "selected": "baseline",
        "baseline": estimator,
        "model": estimator,
    }


def test_predict_outcome_returns_normalized_probs() -> None:
    bundle = _toy_bundle()
    features = {col: 0.0 for col in dataset.FEATURE_COLUMNS}
    probs = predict.predict_outcome(bundle, features)
    assert set(probs) == {"H", "D", "A"}
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)


def test_production_estimator_picks_selected() -> None:
    bundle = _toy_bundle()
    bundle["selected"] = "baseline"
    assert predict.production_estimator(bundle) is bundle["baseline"]
    bundle["selected"] = "xgboost"
    assert predict.production_estimator(bundle) is bundle["model"]


def test_predict_handles_nan_features() -> None:
    bundle = _toy_bundle()
    features = {col: float("nan") for col in dataset.FEATURE_COLUMNS}
    probs = predict.predict_outcome(bundle, features)  # imputer handles NaN
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)
