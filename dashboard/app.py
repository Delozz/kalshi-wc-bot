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
from execution.portfolio import PortfolioState, sync_from_kalshi

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
        row = _analysis_for_ticker(pos.ticker)
        if row is not None:
            # Full thesis from the analysis row (model/book/Kalshi breakdown); the bet
            # size still comes from the signal that sized it.
            theses[pos.ticker] = explain_analysis(
                row, bet_size_cents=(sig or {}).get("bet_size_cents")
            )
        elif sig:
            # Pre-board bets: reduced thesis from the stored signal fields only.
            theses[pos.ticker] = explain_signal(sig)
    return theses


def _latest_board() -> list[dict[str, Any]]:
    """The latest signal cycle's fixture-analysis rows, or [] on any failure (L9).

    Sorted for display: fixtures with the largest absolute blended-vs-Kalshi edge first
    (the discrepancies worth a human look float to the top), legs in H/D/A order within
    a fixture, no-market rows last.
    """
    try:
        from data.db import connect, latest_analysis

        with connect() as conn:
            rows = [dict(row) for row in latest_analysis(conn)]
    except Exception as exc:  # noqa: BLE001 — dashboard must render even with no DB
        logger.debug("Could not read fixture analysis: %s", exc)
        return []

    def fixture_key(row: dict[str, Any]) -> tuple[str, str]:
        return (str(row["home_team"]), str(row["away_team"]))

    max_edge: dict[tuple[str, str], float] = {}
    for row in rows:
        edge = abs(float(row["edge"] or 0.0))
        key = fixture_key(row)
        max_edge[key] = max(max_edge.get(key, 0.0), edge)

    leg_order = {"H": 0, "D": 1, "A": 2, None: 3}
    rows.sort(
        key=lambda r: (
            -max_edge[fixture_key(r)],
            fixture_key(r),
            leg_order.get(r["leg"], 3),
        )
    )
    return rows


def _analysis_for_ticker(ticker: str) -> dict[str, Any] | None:
    """The analysis row behind a placed bet, or None (pre-feature bets / any failure)."""
    if not ticker:
        return None
    try:
        from data.db import analysis_for_ticker, connect

        with connect() as conn:
            row = analysis_for_ticker(conn, ticker)
        return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001 — dashboard must render even with no DB
        logger.debug("Could not read analysis for %s: %s", ticker, exc)
        return None


def _describe_leg(leg: str | None, home: str, away: str) -> str:
    """Human phrasing of a bet leg: who has to do what for the YES to pay."""
    if leg == "H":
        return f"{home} to beat {away}"
    if leg == "A":
        return f"{away} to beat {home}"
    if leg == "D":
        return f"Draw, {home}-{away}"
    return f"{home} vs {away}"


def explain_analysis(row: dict[str, Any], *, bet_size_cents: int | None = None) -> str:
    """One-line thesis for a bet from its fixture-analysis row.

    Example: "Portugal to beat Ghana: model 55%, books 58%, Kalshi 48c -> +9.5% edge,
    half-Kelly $1.20". The anchor phrasing says where the market side of the blend came
    from — book consensus, or the Kalshi price itself when no book covered the fixture.
    """
    matchup = _describe_leg(
        row.get("leg"), str(row["home_team"]), str(row["away_team"])
    )
    tilted = float(row.get("tilted_prob") or 0.0)
    price = float(row.get("kalshi_price") or 0.0)
    edge = float(row.get("edge") or 0.0)
    if row.get("anchor_source") == "book":
        anchor_txt = f"books {float(row.get('anchor_prob') or 0.0):.0%}"
    else:
        anchor_txt = "no book line (Kalshi-anchored)"
    thesis = (
        f"{matchup}: model {tilted:.0%}, {anchor_txt}, "
        f"Kalshi {price * 100:.0f}c -> {edge:+.1%} edge"
    )
    if bet_size_cents:
        thesis += f", half-Kelly ${bet_size_cents / 100:.2f}"
    return thesis


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
    board: list[dict[str, Any]] | None = None,
    console: Console | None = None,
) -> None:
    """Render the dashboard tables to the console.

    ``position_theses`` maps an open position's ticker to its plain-English bet thesis
    (from ``_position_theses``); tickers absent from it render as a dash. ``board``
    (from ``_latest_board``) is the latest cycle's per-leg model-vs-Kalshi breakdown;
    omitted or empty, the board table is skipped (e.g. before the first analysis cycle).
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

    if board:
        cycle = str(board[0].get("cycle_ts", ""))[:16].replace("T", " ")
        board_table = Table(title=f"FIXTURE BOARD — model vs market, cycle {cycle} UTC")
        for column in (
            "Fixture",
            "Leg",
            "Model",
            "Book",
            "Blend",
            "Kalshi",
            "Edge",
            "Decision",
        ):
            board_table.add_column(
                column,
                justify="right" if column not in ("Fixture", "Decision") else "left",
            )
        last_fixture = None
        for row in board:
            fixture = f"{row['home_team']} vs {row['away_team']}"
            shown = fixture if fixture != last_fixture else ""
            last_fixture = fixture
            if row.get("leg") is None:
                board_table.add_row(
                    shown, "—", "—", "—", "—", "—", "—", "no Kalshi market"
                )
                continue

            def pct(key: str) -> str:
                value = row.get(key)
                return f"{float(value):.0%}" if value is not None else "—"

            price = row.get("kalshi_price")
            edge = row.get("edge")
            book = pct("anchor_prob") if row.get("anchor_source") == "book" else "—"
            board_table.add_row(
                shown,
                str(row["leg"]),
                pct("tilted_prob"),
                book,
                pct("blended_prob"),
                f"{float(price) * 100:.0f}c" if price is not None else "—",
                f"{float(edge):+.1%}" if edge is not None else "—",
                str(row["decision"]),
            )
        console.print(board_table)

    _render_legend(console)


def _legend_table(title: str, rows: list[tuple[str, str]]) -> Table:
    table = Table(title=title, title_justify="left", show_header=False, expand=False)
    table.add_column("Term", style="bold", no_wrap=True)
    table.add_column("Meaning")
    for term, meaning in rows:
        table.add_row(term, meaning)
    return table


def _render_legend(console: Console) -> None:
    """Legend explaining every dashboard field and why the model bets what it bets."""
    console.print(
        _legend_table(
            "LEGEND — HOW A BET HAPPENS",
            [
                (
                    "Pipeline",
                    "Dixon-Coles goals model -> confederation/squad/lineup tilts -> "
                    f"blended {settings.model_blend_weight:.0%} model / "
                    f"{1 - settings.model_blend_weight:.0%} market anchor -> edge vs "
                    "the Kalshi ask. A leg is bet only if edge >= "
                    f"{settings.min_edge_threshold:.0%} AND every guard passes; the "
                    "stake is half-Kelly, capped at "
                    f"{settings.max_bet_fraction:.0%} of bankroll, one bet per "
                    f"fixture, total exposure <= {settings.max_portfolio_exposure:.0%}"
                    ", all betting halts at a "
                    f"{settings.stop_loss_threshold:.0%} drawdown from peak "
                    "(stop-loss).",
                ),
                (
                    "Why anchor?",
                    "18 early bets showed the raw model losing to the market line "
                    "(worse Brier). Anchoring means an edge must come from the books "
                    "disagreeing with Kalshi, or survive the shrink toward the "
                    "market — raw model opinion alone can no longer fire a bet.",
                ),
            ],
        )
    )
    console.print(
        _legend_table(
            "LEGEND — FIXTURE BOARD COLUMNS",
            [
                ("Leg", "Outcome: H home win, D draw, A away win (YES contracts)."),
                (
                    "Model",
                    "The model's own probability after tilts, BEFORE the market "
                    "anchor — 'what our model thinks'.",
                ),
                (
                    "Book",
                    "Sportsbook no-vig consensus (median across bookmakers). '—' "
                    "means no book line; the blend anchors to normalized Kalshi "
                    "prices instead.",
                ),
                (
                    "Blend",
                    f"{settings.model_blend_weight:.0%} Model + "
                    f"{1 - settings.model_blend_weight:.0%} anchor — the probability "
                    "every bet decision actually runs on.",
                ),
                (
                    "Kalshi",
                    "YES ask price in cents = the market's implied probability.",
                ),
                (
                    "Edge",
                    "Blend minus Kalshi price. Positive = the blend thinks the "
                    f"contract is underpriced; needs >= +{settings.min_edge_threshold:.0%} "
                    "to bet.",
                ),
                ("Decision", "What happened to the leg — codes below."),
                (
                    "Sorting",
                    "Fixtures with the largest absolute edge on any leg sort first — "
                    "the discrepancies worth a human look are on top.",
                ),
            ],
        )
    )
    console.print(
        _legend_table(
            "LEGEND — DECISION CODES",
            [
                ("signal", "Bet placed (or would be, when running dry)."),
                (
                    "below_threshold",
                    f"Edge under the {settings.min_edge_threshold:.0%} minimum — "
                    "no bet.",
                ),
                (
                    "filtered:below_price_floor",
                    "Market under 6c: longshot noise, not signal.",
                ),
                (
                    "filtered:model_market_mismatch",
                    "Blend prices the leg far above the line (>2.5x favorite, >1.6x "
                    "draw/underdog) — likelier miscalibration than value; defer to "
                    "the market.",
                ),
                (
                    "filtered:powerhouse_favorite",
                    "Draw/upset leg against a >=200 ELO favorite — the model's upset "
                    "probabilities aren't trusted in mismatches.",
                ),
                (
                    "filtered:powerhouse_favorite_squad",
                    "Same guard at a >=150 ELO gap when squad ratings independently "
                    "confirm the favorite.",
                ),
                ("held", "Already holding this market — never top up a position."),
                (
                    "one_per_fixture",
                    "A higher-edge leg of the same match took the single slot (the "
                    "H/D/A legs are mutually exclusive).",
                ),
                (
                    "risk:*",
                    "Portfolio guard: stop_loss "
                    f"({settings.stop_loss_threshold:.0%} drawdown halt), "
                    "max_positions (10 open), insufficient_liquidity (open interest "
                    "floor), or exposure_cap "
                    f"({settings.max_portfolio_exposure:.0%} of bankroll).",
                ),
                (
                    "no_market",
                    "No Kalshi market matched this fixture — a coverage gap worth "
                    "checking, not a decision.",
                ),
            ],
        )
    )
    console.print(
        _legend_table(
            "LEGEND — PORTFOLIO & THESES",
            [
                ("Bankroll / Peak", "Current Kalshi balance / its high-water mark."),
                (
                    "Exposure",
                    "Cost of all open positions; new bets must keep it <= "
                    f"{settings.max_portfolio_exposure:.0%} of bankroll.",
                ),
                ("Avg cents", "Average entry price per contract on the position."),
                (
                    "Why (thesis)",
                    "'matchup: model X%, books Y%, Kalshi Zc -> +E% edge, half-Kelly "
                    "$B' — the model view vs the books vs the market at bet time, "
                    "and the stake that produced. '(Kalshi-anchored)' flags a bet "
                    "made without a book line.",
                ),
            ],
        )
    )


def main() -> None:
    import asyncio

    from data.db import connect, init_db, real_peak_bankroll
    from execution.portfolio import ratchet_peak

    configure_logging()
    state = asyncio.run(
        sync_from_kalshi(fallback_bankroll_cents=settings.initial_bankroll_cents)
    )
    init_db()
    with connect() as conn:
        ratchet_peak(state, real_peak_bankroll(conn))
    render(
        state,
        _recent_signals(),
        position_theses=_position_theses(state.positions),
        board=_latest_board(),
    )


if __name__ == "__main__":
    main()
