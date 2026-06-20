"""Optuna hyperparameter tuning for XGBoost (model/tune.py).

Tunes on the TRAINING set only via StratifiedKFold cross-validation on mlogloss. It
NEVER touches the validation or holdout set — tuning against held-out data invalidates
the experiment (L2). Returns the best hyperparameters for ``xgboost_model.build_xgboost``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from model.xgboost_model import build_xgboost

logger = logging.getLogger(__name__)


def tune_xgboost(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int = 25,
    cv: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Search XGBoost hyperparameters; return the best by CV log loss."""

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
        pipe = build_xgboost(**params)
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipe, x, y, cv=splitter, scoring="neg_log_loss", n_jobs=1
        )
        return float(-scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "Optuna best CV mlogloss=%.4f over %d trials | params=%s",
        study.best_value,
        n_trials,
        study.best_params,
    )
    return study.best_params
