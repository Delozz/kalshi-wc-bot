"""Model evaluation (model/evaluate.py).

Always reports Brier score, log loss, and calibration — never accuracy alone (PRD
section 6.5). Multiclass Brier is the sum of squared errors across the three outcome
probabilities, averaged over matches (uniform 1/3-1/3-1/3 guessing scores ~0.667).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)

_N_CLASSES = 3
_LABELS = [0, 1, 2]


@dataclass(frozen=True)
class EvalResult:
    n: int
    brier: float
    log_loss: float
    accuracy: float


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean over samples of the summed squared error across the 3 class probabilities."""
    onehot = np.eye(_N_CLASSES)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def evaluate(y_true: np.ndarray, probs: np.ndarray) -> EvalResult:
    """Compute Brier, log loss, and accuracy for predicted class probabilities."""
    preds = probs.argmax(axis=1)
    return EvalResult(
        n=int(len(y_true)),
        brier=multiclass_brier(y_true, probs),
        log_loss=float(log_loss(y_true, probs, labels=_LABELS)),
        accuracy=float((preds == y_true).mean()),
    )


def reliability_curve(
    y_true: np.ndarray, probs: np.ndarray, *, n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """Confidence-calibration curve on the predicted class.

    Returns ``(mean_confidence, empirical_accuracy, count)`` per non-empty bin. A
    well-calibrated model has mean_confidence ~ empirical_accuracy in every bin.
    """
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (preds == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)

    rows: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        mask = bucket == b
        count = int(mask.sum())
        if count:
            rows.append(
                (float(confidence[mask].mean()), float(correct[mask].mean()), count)
            )
    return rows
