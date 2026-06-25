"""Tests for the Dixon-Coles goals model (model/dixon_coles.py).

Correctness on small synthetic data — these run in well under a second; the real
fit-on-20k-matches path is exercised by the backtest, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import dixon_coles as dc


def _synthetic(n_per_pair: int = 40) -> pd.DataFrame:
    """A league where 'Strong' beats 'Weak' heavily and edges 'Mid'; all neutral venues."""
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2015-01-01", tz="UTC")
    scorelines = {
        ("Strong", "Weak"): (3, 0),
        ("Strong", "Mid"): (2, 1),
        ("Mid", "Weak"): (2, 0),
    }
    day = 0
    for (h, a), (gh, ga) in scorelines.items():
        for _ in range(n_per_pair):
            # jitter the goals a little so the MLE has variance to work with
            rows.append(
                {
                    "date": base + pd.Timedelta(days=day),
                    "home_team": h,
                    "away_team": a,
                    "fthg": max(0, gh + int(rng.integers(-1, 2))),
                    "ftag": max(0, ga + int(rng.integers(-1, 2))),
                    "neutral": True,
                }
            )
            day += 1
    return pd.DataFrame(rows)


def test_predict_hda_is_a_distribution() -> None:
    model = dc.fit(_synthetic(), min_matches=3)
    probs = model.predict_hda("Strong", "Weak", neutral=True)
    assert set(probs) == {"H", "D", "A"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_strong_beats_weak() -> None:
    model = dc.fit(_synthetic(), min_matches=3)
    probs = model.predict_hda("Strong", "Weak", neutral=True)
    assert probs["H"] > probs["A"]
    assert probs["H"] > probs["D"]


def test_score_matrix_sums_to_one() -> None:
    model = dc.fit(_synthetic(), min_matches=3)
    matrix = model.score_matrix("Strong", "Mid", neutral=True)
    assert abs(matrix.sum() - 1.0) < 1e-9
    assert matrix.shape == (model.max_goals + 1, model.max_goals + 1)


def test_unknown_team_defaults_to_average() -> None:
    # An unseen team falls back to 0/0 ratings rather than raising.
    model = dc.fit(_synthetic(), min_matches=3)
    probs = model.predict_hda("Strong", "Nowhereland", neutral=True)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["H"] > probs["A"]  # Strong still favoured over an average side


def test_home_advantage_helps_on_non_neutral() -> None:
    model = dc.DixonColesModel(
        attack={"A": 0.0, "B": 0.0},
        defense={"A": 0.0, "B": 0.0},
        home_adv=0.4,
        rho=0.0,
    )
    # Identical teams: at a neutral venue H == A; with home advantage H > A.
    neutral = model.predict_hda("A", "B", neutral=True)
    assert abs(neutral["H"] - neutral["A"]) < 1e-9
    home = model.predict_hda("A", "B", neutral=False)
    assert home["H"] > home["A"]


def test_rho_shifts_draw_mass() -> None:
    # The DC correction with positive rho changes the draw balance vs rho = 0.
    common = dict(
        attack={"A": 0.0, "B": 0.0}, defense={"A": 0.0, "B": 0.0}, home_adv=0.0
    )
    no_corr = dc.DixonColesModel(rho=0.0, **common).predict_hda("A", "B", neutral=True)
    pos_corr = dc.DixonColesModel(rho=0.1, **common).predict_hda("A", "B", neutral=True)
    assert pos_corr["D"] != no_corr["D"]
    assert abs(sum(pos_corr.values()) - 1.0) < 1e-9
