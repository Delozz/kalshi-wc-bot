"""Squad-strength feature (features/squad.py).

Turns a national-team squad's per-player season ratings into a single home-minus-away
strength score in ``[-1, 1]`` that ``strategy/edge.py`` applies as an *always-on*
probability prior. Unlike announced lineups (published only ~1h before kickoff), squad
ratings are available at any bet horizon, so star-laden squads — Portugal with Ronaldo,
Norway with Haaland — are favoured even days out, which is exactly when we place bets.

Strength is the mean of the **top-11** player ratings ("expected starting quality"). The
full-squad mean dilutes stars (every nation averages ~6.9, so it barely separates teams);
the top-4 is too noisy (a single deep squad erases a real gap). Top-11 tracks intuition
while still capping single-player noise.

Zero-impact by design: a side with no rated players yields a ``0.0`` delta, so signals
fall back to the model probability alone (the same graceful-degradation contract as
``features/lineup.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)

# API-Football player ratings are on a 0–10 scale; divide the home-minus-away gap by the
# scale to normalize it into [-1, 1] before edge.py applies it with ``squad_weight``.
_RATING_SCALE = 10.0
_TOP_N = 11  # ~ a starting XI: captures star power without single-player noise


def squad_strength(
    ratings: Mapping[int, float] | None, *, top_n: int = _TOP_N
) -> float | None:
    """Mean of the top-``top_n`` player ratings, or ``None`` if no players are rated.

    Returning ``None`` (rather than 0.0) for an unrated squad lets the delta degrade to a
    zero-impact prior instead of fabricating strength from missing data.
    """
    if not ratings:
        return None
    top = sorted(ratings.values(), reverse=True)[:top_n]
    if not top:
        return None
    return sum(top) / len(top)


def squad_strength_delta(
    home_ratings: Mapping[int, float] | None,
    away_ratings: Mapping[int, float] | None,
    *,
    top_n: int = _TOP_N,
) -> float:
    """Home-minus-away squad strength in ``[-1, 1]`` (0.0 if either side is unrated).

    A positive value means the home squad is the stronger one. Unlike the announced-lineup
    delta, this is *not* signed per outcome — ``strategy/edge.py`` uses it to tilt the full
    {H, D, A} probability vector toward the stronger squad.
    """
    home = squad_strength(home_ratings, top_n=top_n)
    away = squad_strength(away_ratings, top_n=top_n)
    if home is None or away is None:
        return 0.0
    delta = (home - away) / _RATING_SCALE
    return max(-1.0, min(1.0, delta))
