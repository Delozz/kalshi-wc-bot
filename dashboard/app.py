"""Dashboard (dashboard/app.py) — rich CLI view of bankroll, positions, and signals.

Renders the current :class:`PortfolioState` and the most recent signals from SQLite
(PRD section 11). Read-only: it never places orders.

Run: ``python -m dashboard.app``
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from rich.console import Console
from rich.table import Table

from config import configure_logging, settings
from execution.portfolio import PortfolioState

logger = logging.getLogger(__name__)


def _cycle_cutoff(newest_iso: str, minutes: int) -> str:
    """ISO timestamp ``minutes`` before ``newest_iso`` (the start of the latest cycle)."""
    ts = datetime.fromisoformat(newest_iso.replace("Z", "+00:00"))
    return (ts - timedelta(minutes=minutes)).isoformat()


def _recent_signals(
    limit: int = 5, *, cycle_window_minutes: int = 5
) -> list[dict[str, Any]]:
    """Recent signals from the LATEST generation cycle, or [] on any failure (L9).

    Signals are written in clustered per-cycle batches. A global last-N-by-time query
    lets stale rows from earlier runs — possibly sized against a different bankroll —
    leak into the live monitor (e.g. a $5.00 bet shown next to a $54 bankroll). Scoping
    to signals generated within ``cycle_window_minutes`` of the newest one keeps the
    panel to the current cycle, so every bet size shown reflects the bankroll that
    actually sized it.
    """
    try:
        from data.db import connect

        with connect() as conn:
            newest = conn.execute(
                "SELECT MAX(generated_at) AS ts FROM signals"
            ).fetchone()
            if newest is None or newest["ts"] is None:
                return []
            cutoff = _cycle_cutoff(str(newest["ts"]), cycle_window_minutes)
            rows = conn.execute(
                "SELECT market_ticker, model_prob, market_implied, edge, bet_size_cents "
                "FROM signals WHERE generated_at >= ? "
                "ORDER BY generated_at DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:  # noqa: BLE001 — dashboard must render even with no DB
        logger.debug("Could not read signals: %s", exc)
        return []


def render(
    state: PortfolioState,
    signals: list[dict[str, Any]],
    *,
    console: Console | None = None,
) -> None:
    """Render the dashboard tables to the console."""
    console = console or Console()

    summary = Table(title=f"KALSHI WC BOT — {settings.kalshi_env.upper()}")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Bankroll", f"${state.bankroll_cents / 100:,.2f}")
    summary.add_row("Peak", f"${state.peak_bankroll_cents / 100:,.2f}")
    summary.add_row("Open positions", str(state.open_count))
    summary.add_row("Exposure", f"${state.exposure_cents / 100:,.2f}")
    console.print(summary)

    positions = Table(title="OPEN POSITIONS")
    for column in ("Ticker", "Side", "Count", "Avg cents"):
        positions.add_column(column)
    for pos in state.positions:
        positions.add_row(
            pos.ticker, pos.side, str(pos.count), str(pos.avg_price_cents)
        )
    console.print(positions)

    signal_table = Table(title="RECENT SIGNALS")
    for column in ("Market", "Model", "Implied", "Edge", "Bet $"):
        signal_table.add_column(column)
    for sig in signals:
        signal_table.add_row(
            str(sig.get("market_ticker", "")),
            f"{sig.get('model_prob', 0):.2f}",
            f"{sig.get('market_implied', 0):.2f}",
            f"{sig.get('edge', 0):+.2%}",
            f"{sig.get('bet_size_cents', 0) / 100:,.2f}",
        )
    console.print(signal_table)


def main() -> None:
    configure_logging()
    state = PortfolioState(
        bankroll_cents=settings.initial_bankroll_cents,
        peak_bankroll_cents=settings.initial_bankroll_cents,
    )
    render(state, _recent_signals())


if __name__ == "__main__":
    main()
