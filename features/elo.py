"""ELO ratings (features/elo.py).

Sequential ELO replay is inherently causal: a match's pre-match ratings depend only
on prior results, so taking each row's pre-match rating introduces no look-ahead.

Home advantage is applied only to non-neutral matches (via the ``neutral`` flag).
Neutral World Cup matches grant a small bonus to the host nation only. The K-factor
can scale with tournament importance (friendly < qualifier < major final).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

logger = logging.getLogger(__name__)

BASE_RATING: float = 1500.0
HOME_ADVANTAGE: float = 65.0  # ELO points added to the home side at non-neutral venues
HOST_BONUS: float = 35.0  # small bonus to the host nation at neutral WC venues

# K-factors. K_GROUP/K_KNOCKOUT are retained for backward compatibility; the
# international pipeline scales K by tournament importance instead.
K_GROUP: float = 32.0
K_KNOCKOUT: float = 40.0
K_FRIENDLY: float = 20.0
K_MAJOR: float = 40.0
K_DEFAULT: float = 30.0

_RESULT_SCORE: dict[str, float] = {"H": 1.0, "D": 0.5, "A": 0.0}
_MAJOR_TOURNAMENTS: tuple[str, ...] = (
    "fifa world cup",
    "uefa euro",
    "copa américa",
    "copa america",
)


def expected_home(
    home_rating: float, away_rating: float, home_advantage: float = 0.0
) -> float:
    """Expected score for the home team, optionally with a home-advantage bump."""
    return 1.0 / (
        1.0 + 10.0 ** ((away_rating - (home_rating + home_advantage)) / 400.0)
    )


def k_for_tournament(tournament: object) -> float:
    """Pick a K-factor from the tournament name (friendly < qualifier < major)."""
    if not isinstance(tournament, str):
        return K_DEFAULT
    name = tournament.lower()
    if "friendly" in name:
        return K_FRIENDLY
    if "quali" in name:
        return K_DEFAULT
    if any(major in name for major in _MAJOR_TOURNAMENTS):
        return K_MAJOR
    return K_DEFAULT


def venue_adjustment(
    home: str,
    away: str,
    *,
    neutral: bool,
    host: str | None = None,
    home_advantage: float = HOME_ADVANTAGE,
    host_bonus: float = HOST_BONUS,
) -> float:
    """Rating bump applied to the home side for the prediction context.

    Non-neutral: full home advantage. Neutral (e.g. World Cup): zero, except a small
    bonus to the host nation (negative when the host is the nominal away team).
    """
    if not neutral:
        return home_advantage
    if host is None:
        return 0.0
    if home == host:
        return host_bonus
    if away == host:
        return -host_bonus
    return 0.0


def _apply(
    ratings: dict[str, float],
    home: str,
    away: str,
    ftr: object,
    k: float,
    base: float,
    home_adv: float,
) -> None:
    score = _RESULT_SCORE.get(ftr) if isinstance(ftr, str) else None
    if score is None:
        return
    h = ratings.get(home, base)
    a = ratings.get(away, base)
    exp_h = expected_home(h, a, home_adv)
    ratings[home] = h + k * (score - exp_h)
    ratings[away] = a + k * ((1.0 - score) - (1.0 - exp_h))


def _row_k(row: object, k: float, use_tournament_k: bool) -> float:
    if use_tournament_k:
        return k_for_tournament(getattr(row, "tournament", None))
    return k


def _row_home_adv(row: object, has_neutral: bool) -> float:
    if has_neutral:
        return 0.0 if bool(getattr(row, "neutral")) else HOME_ADVANTAGE
    return 0.0


def run_elo(
    matches: pd.DataFrame,
    *,
    base: float = BASE_RATING,
    k: float = K_GROUP,
    use_tournament_k: bool = False,
) -> pd.DataFrame:
    """Replay matches in date order, recording each row's pre-match ratings.

    Returns the date-sorted frame with ``home_elo_pre``, ``away_elo_pre`` and
    ``elo_delta`` columns. Per-row pre-match values are causal (no look-ahead). If the
    frame has a ``neutral`` column, home advantage is applied only to non-neutral rows.
    """
    ordered = matches.sort_values("date").reset_index(drop=True)
    has_neutral = "neutral" in ordered.columns
    ratings: dict[str, float] = {}
    home_pre: list[float] = []
    away_pre: list[float] = []
    for row in ordered.itertuples(index=False):
        home_pre.append(ratings.get(row.home_team, base))
        away_pre.append(ratings.get(row.away_team, base))
        _apply(
            ratings,
            row.home_team,
            row.away_team,
            row.ftr,
            _row_k(row, k, use_tournament_k),
            base,
            _row_home_adv(row, has_neutral),
        )
    ordered["home_elo_pre"] = home_pre
    ordered["away_elo_pre"] = away_pre
    ordered["elo_delta"] = ordered["home_elo_pre"] - ordered["away_elo_pre"]
    return ordered


def final_ratings(
    matches: pd.DataFrame,
    *,
    base: float = BASE_RATING,
    k: float = K_GROUP,
    use_tournament_k: bool = False,
) -> dict[str, float]:
    """Ratings after processing all (already cutoff-filtered) matches."""
    ordered = matches.sort_values("date")
    has_neutral = "neutral" in ordered.columns
    ratings: dict[str, float] = {}
    for row in ordered.itertuples(index=False):
        _apply(
            ratings,
            row.home_team,
            row.away_team,
            row.ftr,
            _row_k(row, k, use_tournament_k),
            base,
            _row_home_adv(row, has_neutral),
        )
    return ratings


def elo_delta(
    ratings: Mapping[str, float], home: str, away: str, *, base: float = BASE_RATING
) -> float:
    """Home minus away rating, defaulting unseen teams to ``base``."""
    return ratings.get(home, base) - ratings.get(away, base)
