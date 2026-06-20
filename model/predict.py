"""Model inference (model/predict.py).

Loads the production model artifact and turns a feature dict into calibrated
[P_home, P_draw, P_away] probabilities. The artifact records which model was selected
(``selected``) — the baseline unless XGBoost actually beat it. Calibration is baked into
the saved pipelines, so probabilities are ready for Kelly sizing (L5).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

OUTCOME_ORDER: tuple[str, str, str] = ("H", "D", "A")  # labels 0, 1, 2


def load_bundle(path: Path | None = None) -> dict[str, Any] | None:
    """Load the newest (or a specific) model artifact; None if none exists."""
    if path is None:
        artifacts = sorted(Path(ARTIFACTS_DIR).glob("model_*.pkl"))
        if not artifacts:
            logger.error("No model artifact found; run `python -m model.train` first")
            return None
        path = artifacts[-1]
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    logger.info(
        "Loaded model artifact %s (selected=%s)", path.name, bundle.get("selected")
    )
    return bundle


def production_estimator(bundle: dict[str, Any]) -> Any:
    """Return the estimator to predict with — the selected production model."""
    if bundle.get("selected") == "xgboost":
        return bundle["model"]
    return bundle["baseline"]


def predict_outcome(
    bundle: dict[str, Any], features: dict[str, float]
) -> dict[str, float]:
    """Predict ``{"H": p, "D": p, "A": p}`` for one match's feature dict."""
    columns = bundle["feature_columns"]
    row = np.array([[float(features[col]) for col in columns]], dtype=float)
    estimator = production_estimator(bundle)
    probs = estimator.predict_proba(row)[0]
    return {OUTCOME_ORDER[i]: float(probs[i]) for i in range(len(OUTCOME_ORDER))}
