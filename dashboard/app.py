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
                "SELECT market_ticker, match_id, model_prob, market_implied, edge, "
                "bet_size_cents FROM signals WHERE generated_at >= ? "
                "ORDER BY generated_at DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:  # noqa: BLE001 — dashboard must render even with no DB
        logger.debug("Could not read signals: %s", exc)
        return []


def _signal_for_ticker(ticker: str) -> dict[str, Any] | None:
    """Most recent stored signal for a market ticker, or None on any failure (L9).

    Used to explain an *open position*: the position table only knows the ticker, so we
    pull the latest signal row that targeted it to recover the model/market/edge thesis.
    """
    if not ticker:
        return None
    try:
        from data.db import connect

        with connect() as conn:
            row = conn.execute(
                "SELECT market_ticker, match_id, model_prob, market_implied, edge, "
                "bet_size_cents FROM signals WHERE market_ticker = ? "
                "ORDER BY generated_at DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001 — dashboard must render even with no DB
        logger.debug("Could not read signal for %s: %s", ticker, exc)
        return None


def _position_theses(positions: list[Any]) -> dict[str, str]:
    """Map each open position's ticker to its bet thesis from the originating signal.

    Resolved here (not in ``render``) so the renderer stays a pure display function, the
    same split already used for ``_recent_signals``. Tickers with no stored signal are
    simply omitted; the renderer shows a dash for those.
    """
    theses: dict[str, str] = {}
    for pos in positions:
        sig = _signal_for_ticker(pos.ticker)
        if sig:
            theses[pos.ticker] = explain_signal(sig)
    return theses


def _decode_match(match_id: str) -> str | None:
    """Turn a ``"{fixture}:{outcome}:{home}_{away}"`` match_id into a readable bet thesis.

    Returns e.g. "Senegal to beat Norway" (A), "France to beat Iraq" (H), or
    "Draw, France-Iraq" (D); None if the match_id isn't in the expected shape.
    """
    parts = match_id.split(":")
    if len(parts) < 3:
        return None
    outcome, teams = parts[1], parts[2].split("_", 1)
    if len(teams) != 2:
        return None
    home, away = teams
    if outcome == "H":
        return f"{home} to beat {away}"
    if outcome == "A":
        return f"{away} to beat {home}"
    if outcome == "D":
        return f"Draw, {home}-{away}"
    return None


def explain_signal(sig: dict[str, Any]) -> str:
    """Plain-English thesis for a bet, derived entirely from the stored signal row.

    Example: "Senegal to beat Norway: model 38% vs market 30% (+8.1% edge), half-Kelly".
    Falls back to a team-less phrasing when the match_id can't be decoded.
    """
    model = float(sig.get("model_prob") or 0.0)
    implied = float(sig.get("market_implied") or 0.0)
    edge = float(sig.get("edge") or 0.0)
    thesis = f"model {model:.0%} vs market {implied:.0%} ({edge:+.1%} edge), half-Kelly"
    matchup = _decode_match(str(sig.get("match_id", "")))
    return f"{matchup}: {thesis}" if matchup else thesis


def render(
    state: PortfolioState,
    signals: list[dict[str, Any]],
    *,
    position_theses: dict[str, str] | None = None,
    console: Console | None = None,
) -> None:
    """Render the dashboard tables to the console.

    ``position_theses`` maps an open position's ticker to its plain-English bet thesis
    (from ``_position_theses``); tickers absent from it render as a dash.
    """
    console = console or Console()
    position_theses = position_theses or {}

    summary = Table(title=f"KALSHI WC BOT — {settings.kalshi_env.upper()}")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Bankroll", f"${state.bankroll_cents / 100:,.2f}")
    summary.add_row("Peak", f"${state.peak_bankroll_cents / 100:,.2f}")
    summary.add_row("Open positions", str(state.open_count))
    summary.add_row("Exposure", f"${state.exposure_cents / 100:,.2f}")
    console.print(summary)

    positions = Table(title="OPEN POSITIONS")
    for column in ("Ticker", "Side", "Count", "Avg cents", "Why"):
        positions.add_column(column)
    for pos in state.positions:
        positions.add_row(
            pos.ticker,
            pos.side,
            str(pos.count),
            str(pos.avg_price_cents),
            position_theses.get(pos.ticker, "—"),
        )
    console.print(positions)

    signal_table = Table(title="RECENT SIGNALS")
    signal_table.add_column("Market")
    signal_table.add_column("Bet $", justify="right")
    signal_table.add_column("Why")
    for sig in signals:
        signal_table.add_row(
            str(sig.get("market_ticker", "")),
            f"{sig.get('bet_size_cents', 0) / 100:,.2f}",
            explain_signal(sig),
        )
    console.print(signal_table)


def main() -> None:
    configure_logging()
    state = PortfolioState(
        bankroll_cents=settings.initial_bankroll_cents,
        peak_bankroll_cents=settings.initial_bankroll_cents,
    )
    render(
        state,
        _recent_signals(),
        position_theses=_position_theses(state.positions),
    )


if __name__ == "__main__":
    main()
