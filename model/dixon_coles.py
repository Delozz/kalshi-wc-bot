"""Dixon-Coles goals model (model/dixon_coles.py).

A scoreline model: each team carries an attack and a defense rating, and a match's goals
are Poisson with means

    log lambda_home = home_adv*(not neutral) + attack_home + defense_away
    log mu_away     =                          attack_away + defense_home

The Dixon-Coles (1997) low-score correction ``rho`` adjusts the four 0-0/1-0/0-1/1-1 cells
where independent Poissons misprice tight games. From the resulting score matrix we read
H/D/A natively: P(home) is the upper triangle, P(draw) the diagonal, P(away) the lower.

Why this exists: the classifier treats "draw" as an arbitrary third label and was shown
(2018 dev + 2022 holdout) to be overconfident in the 45-55% band — exactly the near-coin-flip
games where phantom edge is born. Modelling the diagonal of a score matrix gives the draw an
honest, structurally-derived probability instead.

Fitting is a ridge-penalized, time-decayed Poisson MLE over historical scorelines. Causality
is the caller's responsibility: pass only matches strictly before the cutoff (the backtest
routes them through ``lookahead_guard`` first, L1). Home advantage is learned from
non-neutral matches and set to 0 for neutral World Cup fixtures, mirroring the ELO venue
logic so the two engines stay consistent.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

DEFAULT_MAX_GOALS = 10
# Time-decay rate per day. ~0.0019 puts a half-life near one year, so a decade-old friendly
# barely counts while recent form dominates (Dixon-Coles use a similar weekly xi).
DEFAULT_XI = 0.0019
DEFAULT_RIDGE = (
    1.0  # L2 pull on attack/defense — pins the overall level, stabilises the fit
)
_TAU_FLOOR = 1e-12  # clamp the DC correction so log() never sees a non-positive value
# Validated strength-prior weight (2018 out-of-sample ELO sweep). 0.5 sits at the knee:
# it beats both plain DC (Brier 0.583) and the classifier (0.582) at 0.573 with peak
# accuracy, before higher values trade accuracy/log-loss away by collapsing the goals model
# into ELO-as-Poisson. The live squad-strength prior should reuse this as its default.
DEFAULT_PRIOR_SCALE = 0.5


@dataclass(frozen=True)
class DixonColesModel:
    """Fitted attack/defense ratings plus home advantage and the DC ``rho`` correction."""

    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    max_goals: int = DEFAULT_MAX_GOALS
    _factorials: np.ndarray | None = field(default=None, repr=False, compare=False)

    def _fact(self) -> np.ndarray:
        if self._factorials is None:
            fact = np.array(
                [float(math.factorial(k)) for k in range(self.max_goals + 1)]
            )
            object.__setattr__(self, "_factorials", fact)
        return self._factorials

    def rates(self, home: str, away: str, *, neutral: bool) -> tuple[float, float]:
        """Expected (home_goals, away_goals) for the fixture."""
        ah = self.attack.get(home, 0.0)
        aa = self.attack.get(away, 0.0)
        dh = self.defense.get(home, 0.0)
        da = self.defense.get(away, 0.0)
        adv = 0.0 if neutral else self.home_adv
        lam = float(np.exp(adv + ah + da))
        mu = float(np.exp(aa + dh))
        return lam, mu

    def score_matrix(self, home: str, away: str, *, neutral: bool) -> np.ndarray:
        """Probability matrix ``M[x, y]`` of home scoring x, away scoring y (DC-corrected)."""
        lam, mu = self.rates(home, away, neutral=neutral)
        ks = np.arange(self.max_goals + 1)
        fact = self._fact()
        # Independent Poisson pmfs, outer-product into a score matrix.
        home_pmf = np.exp(-lam) * lam**ks / fact
        away_pmf = np.exp(-mu) * mu**ks / fact
        matrix = np.outer(home_pmf, away_pmf)
        # Dixon-Coles low-score correction on the four tight-game cells.
        matrix[0, 0] *= 1.0 - lam * mu * self.rho
        matrix[0, 1] *= 1.0 + lam * self.rho
        matrix[1, 0] *= 1.0 + mu * self.rho
        matrix[1, 1] *= 1.0 - self.rho
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    def predict_hda(self, home: str, away: str, *, neutral: bool) -> dict[str, float]:
        """Native H/D/A probabilities from the score matrix (lower / diagonal / upper).

        With rows = home goals and cols = away goals, ``home_goals > away_goals`` sits below
        the main diagonal (``np.tril(-1)``), draws on it, and away wins above it.
        """
        matrix = self.score_matrix(home, away, neutral=neutral)
        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())
        return {"H": home_win, "D": draw, "A": away_win}


def _prepare(
    matches: pd.DataFrame, *, cutoff: pd.Timestamp, xi: float, min_matches: int
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Index teams and pack the per-match arrays the likelihood needs."""
    df = matches.dropna(subset=["fthg", "ftag"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    keep = set(counts[counts >= min_matches].index)
    df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]
    if df.empty:
        raise ValueError("No matches left after the min_matches filter; lower it.")

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {team: i for i, team in enumerate(teams)}
    home_i = df["home_team"].map(idx).to_numpy(dtype=int)
    away_i = df["away_team"].map(idx).to_numpy(dtype=int)
    x = df["fthg"].to_numpy(dtype=float)
    y = df["ftag"].to_numpy(dtype=float)
    neutral = (
        df["neutral"].to_numpy(dtype=bool)
        if "neutral" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    days = (cutoff - df["date"]).dt.total_seconds().to_numpy() / 86400.0
    weights = np.exp(-xi * np.clip(days, 0.0, None))
    return teams, home_i, away_i, x, y, neutral, weights


def _neg_log_likelihood(
    params: np.ndarray,
    n: int,
    home_i: np.ndarray,
    away_i: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    neutral: np.ndarray,
    weights: np.ndarray,
    ridge: float,
    prior_attack: np.ndarray,
    prior_defense: np.ndarray,
) -> float:
    """Weighted DC negative log-likelihood with an L2 pull toward the strength prior.

    The ridge shrinks each team's attack/defense toward ``prior_attack``/``prior_defense``
    (the external strength prior) rather than toward zero. Data-rich teams override the
    prior; sparse teams lean on it. With a zero prior this is ordinary ridge-to-average.
    """
    attack = params[:n]
    defense = params[n : 2 * n]
    home_adv = params[2 * n]
    rho = params[2 * n + 1]

    log_lam = attack[home_i] + defense[away_i] + home_adv * (~neutral)
    log_mu = attack[away_i] + defense[home_i]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    # Poisson log-likelihood (the constant log(x!) term is dropped — irrelevant to argmin).
    ll = x * log_lam - lam + y * log_mu - mu

    # Dixon-Coles low-score correction, vectorised over the four tight-game cells.
    tau = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    ll += np.log(np.clip(tau, _TAU_FLOOR, None))

    penalty = ridge * (
        np.sum((attack - prior_attack) ** 2) + np.sum((defense - prior_defense) ** 2)
    )
    return float(-np.sum(weights * ll) + penalty)


def _strength_prior(
    teams: list[str],
    strength: Mapping[str, float] | None,
    prior_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn an external per-team strength signal into (prior_attack, prior_defense).

    The strength values (e.g. ELO, or squad rating for live) are z-scored across the
    modelled teams, then mapped so a stronger team gets higher attack and *more negative*
    defense (concedes fewer): ``prior_attack = scale*z``, ``prior_defense = -scale*z``. The
    prior is mean-zero by construction, so it shifts where ratings shrink *to* without
    moving the overall level. Returns zero arrays when no strength is supplied — identical
    to the original shrink-to-average behaviour.
    """
    n = len(teams)
    if not strength or prior_scale == 0.0:
        return np.zeros(n), np.zeros(n)
    raw = np.array([float(strength.get(team, np.nan)) for team in teams])
    seen = ~np.isnan(raw)
    if seen.sum() < 2:
        return np.zeros(n), np.zeros(n)
    mean = raw[seen].mean()
    std = raw[seen].std()
    z = np.where(seen, (raw - mean) / std, 0.0) if std > 0 else np.zeros(n)
    prior_attack = prior_scale * z
    prior_defense = -prior_scale * z
    return prior_attack, prior_defense


def fit(
    matches: pd.DataFrame,
    *,
    cutoff: pd.Timestamp | None = None,
    xi: float = DEFAULT_XI,
    ridge: float = DEFAULT_RIDGE,
    min_matches: int = 5,
    max_goals: int = DEFAULT_MAX_GOALS,
    maxiter: int = 1000,
    strength: Mapping[str, float] | None = None,
    prior_scale: float = 0.0,
) -> DixonColesModel:
    """Fit attack/defense/home-adv/rho by time-decayed, ridge-penalised Poisson MLE.

    ``cutoff`` anchors the time-decay (defaults to the latest match date). Callers are
    responsible for passing only causal (pre-kickoff) matches — the backtest seals this with
    ``lookahead_guard`` (L1). Teams with fewer than ``min_matches`` appearances are dropped
    and fall back to average (0/0) ratings at prediction time.

    ``strength`` (optional) is an external per-team strength signal — ELO in the backtest,
    squad rating for live — that the ridge shrinks the ratings *toward* (scaled by
    ``prior_scale``). This lets a star-laden or highly-rated side carry strength the raw
    scorelines under-state, especially for teams with sparse history. With no ``strength``
    the fit is unchanged (shrink toward average).
    """
    if cutoff is None:
        cutoff = pd.to_datetime(matches["date"], utc=True).max()
    cutoff = pd.Timestamp(cutoff)
    cutoff = (
        cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    )

    teams, home_i, away_i, x, y, neutral, weights = _prepare(
        matches, cutoff=cutoff, xi=xi, min_matches=min_matches
    )
    n = len(teams)
    prior_attack, prior_defense = _strength_prior(teams, strength, prior_scale)

    # Init at the prior (not zero): seeds the optimiser near the strength-implied ratings,
    # a mild positive home advantage, zero correlation.
    init = np.zeros(2 * n + 2)
    init[:n] = prior_attack
    init[n : 2 * n] = prior_defense
    init[2 * n] = 0.25
    bounds = [(None, None)] * (2 * n) + [(None, None), (-0.9, 0.9)]

    result = minimize(
        _neg_log_likelihood,
        init,
        args=(
            n,
            home_i,
            away_i,
            x,
            y,
            neutral,
            weights,
            ridge,
            prior_attack,
            prior_defense,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        # maxfun must scale with the parameter count: a single numerical gradient costs
        # ~2*len(params) evaluations, so the default 15k cap starves a ~600-param fit after
        # a couple dozen iterations. Budget generously so the MLE actually converges.
        options={"maxiter": maxiter, "maxfun": 200 * len(init)},
    )
    if not result.success:
        logger.warning("Dixon-Coles fit did not fully converge: %s", result.message)

    params = result.x
    attack = {team: float(params[i]) for i, team in enumerate(teams)}
    defense = {team: float(params[n + i]) for i, team in enumerate(teams)}
    logger.info(
        "Fitted Dixon-Coles on %d matches, %d teams (home_adv=%.3f, rho=%.3f)",
        len(x),
        n,
        float(params[2 * n]),
        float(params[2 * n + 1]),
    )
    return DixonColesModel(
        attack=attack,
        defense=defense,
        home_adv=float(params[2 * n]),
        rho=float(params[2 * n + 1]),
        max_goals=max_goals,
    )
