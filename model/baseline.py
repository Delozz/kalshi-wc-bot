"""Logistic regression baseline (model/baseline.py).

The floor any complex model must beat on Brier score (PRD section 6.3). Multinomial
logistic on the odds-free team-strength features, wrapped in a pipeline that imputes
missing form/H2H values and standardizes before fitting.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline(*, c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """Build the (unfitted) logistic-regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )
