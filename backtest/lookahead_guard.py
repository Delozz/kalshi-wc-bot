"""Look-ahead guard (L1) — the single most important safety net in the backtester.

Every feature computation MUST route its source data through `filter_data` with a
cutoff equal to the match kickoff. Any row dated at or after the cutoff is a
look-ahead violation and raises LookAheadError. Look-ahead bias makes losing
strategies appear profitable; this guard is non-negotiable.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class LookAheadError(Exception):
    """Raised when source data contains rows at or after the cutoff time."""


def filter_data(
    df: pd.DataFrame,
    cutoff: datetime,
    timestamp_col: str = "date",
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Return only rows strictly before ``cutoff``.

    Args:
        df: source frame containing a datetime column.
        cutoff: exclusive upper bound (typically match kickoff, UTC).
        timestamp_col: name of the datetime column to filter on.
        strict: if True, raise :class:`LookAheadError` when any row is at/after the
            cutoff instead of silently dropping it. Keep this True everywhere except
            trusted internal callers that have already filtered.

    Raises:
        KeyError: if ``timestamp_col`` is missing.
        LookAheadError: if ``strict`` and any row has ``timestamp >= cutoff``.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"timestamp column {timestamp_col!r} not in frame")

    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")

    cutoff_ts = pd.Timestamp(cutoff)
    cutoff_ts = (
        cutoff_ts.tz_localize("UTC")
        if cutoff_ts.tzinfo is None
        else cutoff_ts.tz_convert("UTC")
    )

    after_cutoff = ts >= cutoff_ts
    unparseable = ts.isna()

    if after_cutoff.any():
        n = int(after_cutoff.sum())
        msg = (
            f"{n} row(s) at/after cutoff {cutoff_ts.isoformat()} "
            f"in column {timestamp_col!r}"
        )
        if strict:
            raise LookAheadError(msg)
        logger.warning("Look-ahead rows dropped: %s", msg)

    if unparseable.any():
        logger.warning(
            "Dropping %d row(s) with unparseable %r (cannot verify they precede cutoff)",
            int(unparseable.sum()),
            timestamp_col,
        )

    keep = ~(after_cutoff | unparseable)
    return df.loc[keep].copy()
