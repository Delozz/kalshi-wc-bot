"""Lineup-strength feature (features/lineup.py).

Turns an API-Football ``/fixtures/lineups`` payload into a single home-minus-away
strength score in ``[-1, 1]`` that ``strategy/edge.py`` applies as a post-model
probability nudge. Squad strength is the mean per-player ``rating`` (0–10 scale) over the
announced starting XI.

Zero-impact by design (the key property of the lineup-adjustment plan): if either side
has no announced XI or no rated players, the delta is ``0.0`` and signals fall back to
the model probability alone. Lineups are typically released ~1h before kickoff, so for
most of the betting window this returns ``0.0`` and changes nothing.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# API-Football player ratings are on a 0–10 scale; divide the home-minus-away gap by the
# scale to normalize it into [-1, 1] before edge.py multiplies by ``lineup_weight``.
_RATING_SCALE = 10.0


def _avg_rating(team_lineup: dict[str, Any]) -> float | None:
    """Mean starting-XI player rating for one team, or ``None`` if none are rated.

    Reads ``startXI[].player.rating``. Players without a numeric rating are skipped; if
    no player on the side is rated, returns ``None`` so the caller degrades to a 0.0
    delta rather than fabricating strength from incomplete data.
    """
    start_xi = team_lineup.get("startXI") or []
    ratings: list[float] = []
    for entry in start_xi:
        player = (entry or {}).get("player") or {}
        raw = player.get("rating")
        if raw is None:
            continue
        try:
            ratings.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not ratings:
        return None
    return sum(ratings) / len(ratings)


def lineup_strength_delta(
    home_lineup: dict[str, Any] | None, away_lineup: dict[str, Any] | None
) -> float:
    """Home-minus-away starting-XI strength in ``[-1, 1]`` (0.0 if not fully announced).

    A positive value means the home side fielded the stronger XI. The result is signed
    for the *home* outcome; ``signal_gen.py`` flips the sign for the away outcome and
    zeroes it for the draw before passing it to ``edge.build_signal``.
    """
    if not home_lineup or not away_lineup:
        return 0.0
    home_avg = _avg_rating(home_lineup)
    away_avg = _avg_rating(away_lineup)
    if home_avg is None or away_avg is None:
        return 0.0
    delta = (home_avg - away_avg) / _RATING_SCALE
    return max(-1.0, min(1.0, delta))
