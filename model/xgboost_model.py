"""XGBoost multi-class model (model/xgboost_model.py).

Predicts [P_home, P_draw, P_away]. Must beat the logistic baseline on Brier before it
is used in any production backtest (PRD section 6.3). Hyperparameters are conservative
defaults; tune with optuna later. A median imputer keeps the interface identical to
the baseline (XGBoost itself tolerates NaN, but a shared imputer aids calibration CV).
"""

from __future__ import annotations

from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": 4,
}


def build_xgboost(**overrides: Any) -> Pipeline:
    """Build the (unfitted) XGBoost pipeline, with optional hyperparameter overrides."""
    params = {**_DEFAULT_PARAMS, **overrides}
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(**params)),
        ]
    )
