"""Out-of-sample probability backtest (backtest/strategy_backtest.py).

Phase A of strategy validation: train the model on data STRICTLY BEFORE a World Cup and
evaluate its probabilities ON that tournament, out-of-sample. This answers the question
behind the losing bets — are the model's probabilities trustworthy, or is it manufacturing
fake edges? A model that says 70% but is right 50% of the time produces phantom edge
everywhere. Reports Brier, log loss, accuracy, and a calibration (reliability) curve.

Why retrain here instead of loading the deployed artifact: the live artifact is trained on
internationals up to the 2022 seal, so it has already *seen* the 2018 World Cup — scoring
2018 with it would be in-sample and leaky. Retraining on data before each tournament is the
only honest out-of-sample read.

L1 — every training set is routed through ``lookahead_guard`` against the tournament's first
kickoff, so no tournament data can leak into training.

L2 — the 2022 World Cup is the sacred holdout. ``run_probability_backtest`` refuses any year
>= ``HOLDOUT_YEAR`` unless ``allow_holdout=True`` is passed explicitly, which consumes the
one-time evaluation touch. Never tune against a holdout result.

Price-based P&L (ROI / Sharpe / drawdown) is Phase B: this engine is deliberately
price-agnostic, so a historical-odds provider can plug into ``evaluate_split`` later without
re-deriving the causal training split.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import lookahead_guard
from config import configure_logging
from ingestion import international_results
from features import elo
from model import baseline as baseline_mod
from model import calibration as calib_mod
from model import dixon_coles
from model import evaluate as eval_mod
from model import xgboost_model
from model.dataset import (
    FEATURE_COLUMNS,
    LABEL_MAP,
    build_feature_table,
    split_train_val,
)

logger = logging.getLogger(__name__)

HOLDOUT_YEAR = 2022  # 2022+ is the sacred holdout (L2) — touched once, eval only
HOLDOUT_SEAL = "2022-01-01"  # when not consuming the holdout, never load 2022+ data
TRAIN_START = "2000-01-01"  # PRD: train on internationals from 2000 onward


@dataclass(frozen=True)
class TournamentEval:
    """Out-of-sample evaluation of one tournament."""

    year: int
    n: int
    n_train: int
    baseline: eval_mod.EvalResult
    xgboost: eval_mod.EvalResult
    selected: str  # "baseline" or "xgboost" — whichever wins on out-of-sample Brier
    reliability: list[tuple[float, float, int]]


def _xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame["label"].to_numpy(dtype=int)
    return x, y


def evaluate_split(year: int, train: pd.DataFrame, val: pd.DataFrame) -> TournamentEval:
    """Train on ``train``, evaluate probabilities on ``val`` (the tournament), causally.

    The training frame is first routed through ``lookahead_guard`` against the tournament's
    first kickoff (L1): any training row dated at/after that kickoff raises and aborts the
    backtest rather than silently leaking. Trains the logistic baseline and a calibrated
    XGBoost (L5), and selects whichever wins on out-of-sample Brier.
    """
    if val.empty:
        raise SystemExit(f"No {year} World Cup validation matches found; aborting.")

    # L1 tripwire: prove no training row reaches the tournament's first kickoff.
    cutoff = pd.to_datetime(val["date"]).min().to_pydatetime()
    lookahead_guard.filter_data(train, cutoff)

    x_train, y_train = _xy(train)
    x_val, y_val = _xy(val)

    baseline = baseline_mod.build_baseline()
    baseline.fit(x_train, y_train)
    base_result = eval_mod.evaluate(y_val, baseline.predict_proba(x_val))

    calibrated = calib_mod.calibrate(xgboost_model.build_xgboost(), x_train, y_train)
    xgb_probs = calibrated.predict_proba(x_val)
    xgb_result = eval_mod.evaluate(y_val, xgb_probs)

    selected = "xgboost" if xgb_result.brier < base_result.brier else "baseline"
    selected_probs = (
        xgb_probs if selected == "xgboost" else baseline.predict_proba(x_val)
    )
    reliability = eval_mod.reliability_curve(y_val, selected_probs)

    return TournamentEval(
        year=year,
        n=int(len(y_val)),
        n_train=int(len(y_train)),
        baseline=base_result,
        xgboost=xgb_result,
        selected=selected,
        reliability=reliability,
    )


@dataclass(frozen=True)
class GoalsEval:
    """Out-of-sample Dixon-Coles evaluation, carrying the raw predictions.

    ``probs``/``labels`` are exposed so a calibration layer (temperature scaling) can be
    fit on these genuinely out-of-sample predictions — never on training rows.
    """

    result: eval_mod.EvalResult
    reliability: list[tuple[float, float, int]]
    probs: np.ndarray
    labels: np.ndarray


def evaluate_goals_model(
    year: int, train: pd.DataFrame, val: pd.DataFrame, *, prior_scale: float = 0.0
) -> GoalsEval:
    """Fit Dixon-Coles on ``train`` scorelines and score H/D/A on ``val``, causally.

    Same causal contract as :func:`evaluate_split` — the training frame is routed through
    ``lookahead_guard`` against the tournament's first kickoff (L1) before the model sees
    it. Probabilities are read natively from each fixture's score matrix and scored with the
    shared Brier / log-loss / reliability metrics, so the result is directly comparable to
    the classifier's.

    ``prior_scale`` > 0 turns on the strength prior: the model's attack/defense are shrunk
    toward ELO computed causally on ``train`` (the same ELO the live pipeline uses), so
    sparse-history teams inherit ELO's broader signal. This is the backtestable proxy for
    the live squad-strength prior, which plugs into the identical ``strength`` interface.
    """
    if val.empty:
        raise SystemExit(f"No {year} World Cup validation matches found; aborting.")

    cutoff = pd.to_datetime(val["date"]).min().to_pydatetime()
    lookahead_guard.filter_data(train, cutoff)

    # ELO over the (causal) training matches is the strength prior; 0 scale = plain DC.
    strength = (
        elo.final_ratings(train, use_tournament_k=True) if prior_scale > 0 else None
    )
    model = dixon_coles.fit(
        train,
        cutoff=pd.Timestamp(cutoff),
        strength=strength,
        prior_scale=prior_scale,
    )

    # Column order must match LABEL_MAP (H=0, D=1, A=2) so evaluate() lines probs up
    # with the integer labels.
    order = [outcome for outcome, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]
    probs = np.empty((len(val), 3), dtype=float)
    for i, row in enumerate(val.itertuples(index=False)):
        hda = model.predict_hda(
            row.home_team, row.away_team, neutral=bool(getattr(row, "neutral", True))
        )
        probs[i] = [hda[outcome] for outcome in order]

    y_val = val["label"].to_numpy(dtype=int)
    result = eval_mod.evaluate(y_val, probs)
    reliability = eval_mod.reliability_curve(y_val, probs)
    return GoalsEval(result=result, reliability=reliability, probs=probs, labels=y_val)


def run_probability_backtest(
    year: int = 2018, *, allow_holdout: bool = False
) -> TournamentEval:
    """Causally train-then-evaluate the model on the ``year`` World Cup, out-of-sample.

    For pre-holdout years the data is sealed at ``HOLDOUT_SEAL`` so 2022+ is never even
    loaded. Evaluating ``HOLDOUT_YEAR`` or later requires ``allow_holdout=True`` (L2) — the
    single sanctioned touch of the holdout; never call it during tuning.
    """
    if year >= HOLDOUT_YEAR and not allow_holdout:
        raise SystemExit(
            f"{year} is the sacred holdout (L2). Pass allow_holdout=True to consume the "
            "one-time evaluation touch — and never tune against the result."
        )

    # Seal 2022+ out entirely for pre-holdout backtests; load full history only when
    # deliberately consuming the holdout.
    max_date = None if year >= HOLDOUT_YEAR else HOLDOUT_SEAL
    matches = international_results.load(max_date=max_date)
    if matches.empty:
        raise SystemExit("No international results loaded; aborting.")

    table = build_feature_table(matches).dropna(subset=["label"])
    train, val = split_train_val(table, val_year=year)
    train = train[train["date"] >= pd.Timestamp(TRAIN_START)]
    return evaluate_split(year, train, val)


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
    parser = argparse.ArgumentParser(
        description="Out-of-sample probability backtest for a World Cup year."
    )
    parser.add_argument(
        "--year", type=int, default=2018, help="World Cup year to evaluate (dev=2018)."
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Consume the one-time 2022 holdout touch (L2). Never use during tuning.",
    )
    parser.add_argument(
        "--model",
        choices=("classifier", "dc"),
        default="classifier",
        help="Which engine to evaluate: classifier (baseline+XGBoost) or dc (Dixon-Coles).",
    )
    parser.add_argument(
        "--dc-prior-scale",
        type=float,
        default=0.0,
        help="Strength of the ELO prior for --model dc (0 = plain DC; try 0.1-0.4).",
    )
    parser.add_argument(
        "--fit-dc-temperature",
        action="store_true",
        help=(
            "After a --model dc run, fit the temperature exponent tau on the tournament's "
            "out-of-sample predictions (L5) and report the tempered Brier/log-loss. The "
            "fitted tau is what DC_TEMPERATURE should carry in live config."
        ),
    )
    args = parser.parse_args()
    configure_logging()

    if args.year >= HOLDOUT_YEAR and not args.allow_holdout:
        raise SystemExit(
            f"{args.year} is the sacred holdout (L2). Pass --allow-holdout to consume the "
            "one-time evaluation touch — and never tune against the result."
        )

    # Both engines share the same causal split + holdout seal, so they are comparable.
    max_date = None if args.year >= HOLDOUT_YEAR else HOLDOUT_SEAL
    matches = international_results.load(max_date=max_date)
    if matches.empty:
        raise SystemExit("No international results loaded; aborting.")
    table = build_feature_table(matches).dropna(subset=["label"])
    train, val = split_train_val(table, val_year=args.year)
    train = train[train["date"] >= pd.Timestamp(TRAIN_START)]

    if args.model == "dc":
        goals_eval = evaluate_goals_model(
            args.year, train, val, prior_scale=args.dc_prior_scale
        )
        result, reliability = goals_eval.result, goals_eval.reliability
        logger.info(
            "=== %d World Cup out-of-sample (Dixon-Coles, prior_scale=%.2f, "
            "%d train / %d eval) ===",
            args.year,
            args.dc_prior_scale,
            len(train),
            result.n,
        )
        _log_eval("Dixon-Coles", result)
        if args.fit_dc_temperature:
            tau = calib_mod.fit_temperature(goals_eval.probs, goals_eval.labels)
            tempered = calib_mod.temper_matrix(goals_eval.probs, tau)
            tempered_result = eval_mod.evaluate(goals_eval.labels, tempered)
            _log_eval(f"DC tempered t={tau:.3f}", tempered_result)
            logger.info(
                "Set DC_TEMPERATURE=%.3f in .env if the tempered Brier improves; "
                "keep 1.0 (identity) otherwise.",
                tau,
            )
    else:
        evaluation = evaluate_split(args.year, train, val)
        logger.info(
            "=== %d World Cup out-of-sample (%d train / %d eval matches) ===",
            evaluation.year,
            evaluation.n_train,
            evaluation.n,
        )
        _log_eval("Logistic baseline", evaluation.baseline)
        _log_eval("XGBoost+calibrated", evaluation.xgboost)
        logger.info("Selected (better Brier): %s", evaluation.selected)
        result, reliability = (
            (evaluation.baseline, evaluation.reliability)
            if evaluation.selected == "baseline"
            else (evaluation.xgboost, evaluation.reliability)
        )

    # A uniform 1/3-1/3-1/3 guesser scores Brier ~0.667; that is the bar to clear.
    logger.info("Calibration (predicted-class confidence vs empirical accuracy):")
    for mean_conf, emp_acc, count in reliability:
        logger.info("  conf=%.2f  acc=%.2f  n=%d", mean_conf, emp_acc, count)


if __name__ == "__main__":
    main()
