"""Training entrypoint (model/train.py) — odds-free national-team model.

Pipeline: load international results (2022 holdout SEALED via ``max_date``) -> build
causal features -> train logistic baseline and XGBoost (optionally optuna-tuned) ->
calibrate (L5: before any sizing) -> evaluate on the 2018 WC validation set with
Brier / log loss / calibration. A timestamped artifact is saved to model/artifacts/.

Run: ``python -m model.train``                 (defaults, untuned XGBoost)
     ``python -m model.train --tune 25``       (optuna-tune XGBoost on train CV)
The dev validation set is the 2018 WC; 2022 is never touched.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, configure_logging, ensure_dirs
from ingestion import international_results
from model import baseline as baseline_mod
from model import calibration as calib_mod
from model import evaluate as eval_mod
from model import tune as tune_mod
from model import xgboost_model
from model.dataset import FEATURE_COLUMNS, build_feature_table, split_train_val

logger = logging.getLogger(__name__)

HOLDOUT_SEAL = "2022-01-01"  # never load 2022 WC data during development (L2)
TRAIN_START = "2000-01-01"  # PRD: train on internationals from 2000 onward


def _xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame["label"].to_numpy(dtype=int)
    return x, y


def _log_eval(name: str, result: eval_mod.EvalResult) -> None:
    logger.info(
        "%-20s | Brier=%.4f  LogLoss=%.4f  Acc=%.3f  (n=%d)",
        name,
        result.brier,
        result.log_loss,
        result.accuracy,
        result.n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the odds-free WC model.")
    parser.add_argument(
        "--val-year", type=int, default=2018, help="WC validation year (dev=2018)."
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=0,
        metavar="N_TRIALS",
        help="Optuna trials for XGBoost (0 = use defaults, no tuning).",
    )
    args = parser.parse_args()
    if args.val_year == 2022:
        raise SystemExit(
            "2022 is the sacred holdout (L2). Refusing to train against it."
        )

    configure_logging()
    ensure_dirs()

    matches = international_results.load(max_date=HOLDOUT_SEAL)
    if matches.empty:
        raise SystemExit("No international results loaded; aborting.")

    table = build_feature_table(matches).dropna(subset=["label"])
    train, val = split_train_val(table, val_year=args.val_year)
    train = train[train["date"] >= pd.Timestamp(TRAIN_START)]
    if val.empty:
        raise SystemExit(f"No {args.val_year} WC validation matches found; aborting.")

    x_train, y_train = _xy(train)
    x_val, y_val = _xy(val)
    logger.info(
        "Training on %d matches; validating on %d; %d features.",
        len(x_train),
        len(x_val),
        len(FEATURE_COLUMNS),
    )

    # Uniform reference (1/3 each) — the dumbest possible predictor.
    uniform = np.full((len(y_val), 3), 1.0 / 3.0)
    _log_eval("Uniform reference", eval_mod.evaluate(y_val, uniform))

    # Logistic baseline.
    baseline = baseline_mod.build_baseline()
    baseline.fit(x_train, y_train)
    base_result = eval_mod.evaluate(y_val, baseline.predict_proba(x_val))
    _log_eval("Logistic baseline", base_result)

    # XGBoost (optionally optuna-tuned on train CV) + calibration (L5).
    if args.tune > 0:
        logger.info(
            "Tuning XGBoost with optuna (%d trials, train CV only)...", args.tune
        )
        best_params = tune_mod.tune_xgboost(x_train, y_train, n_trials=args.tune)
        xgb = xgboost_model.build_xgboost(**best_params)
    else:
        xgb = xgboost_model.build_xgboost()
    calibrated = calib_mod.calibrate(xgb, x_train, y_train)
    xgb_probs = calibrated.predict_proba(x_val)
    xgb_result = eval_mod.evaluate(y_val, xgb_probs)
    _log_eval("XGBoost+calibrated", xgb_result)

    beats = xgb_result.brier < base_result.brier
    logger.info(
        "XGBoost beats baseline on Brier: %s (%.4f vs %.4f)",
        beats,
        xgb_result.brier,
        base_result.brier,
    )
    best_name = "xgboost" if beats else "baseline"
    logger.info("Selected production model: %s", best_name)

    logger.info("Calibration (predicted-class confidence vs empirical accuracy):")
    for mean_conf, emp_acc, count in eval_mod.reliability_curve(y_val, xgb_probs):
        logger.info("  conf=%.2f  acc=%.2f  n=%d", mean_conf, emp_acc, count)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = {
        "created_utc": stamp,
        "feature_columns": FEATURE_COLUMNS,
        "val_year": args.val_year,
        "selected": best_name,
        "baseline": baseline,
        "model": calibrated,
        "val_metrics": {
            "baseline": base_result.__dict__,
            "xgboost": xgb_result.__dict__,
        },
    }
    artifact_path = ARTIFACTS_DIR / f"model_{stamp}.pkl"
    with open(artifact_path, "wb") as handle:
        pickle.dump(artifact, handle)
    logger.info("Saved artifact -> %s", artifact_path)


if __name__ == "__main__":
    main()
