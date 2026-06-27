"""Live performance scorecard (dashboard/scorecard.py).

Reads SETTLED orders from SQLite and reports whether the model is actually working on real
money — calibration, hit rate, ROI, and (the decisive question) whether the model's
probabilities beat the Kalshi line on the bets we actually placed. Read-only; never trades.

The honest caveat baked into the design: every metric here is computed over the *selected*
sample of outcomes we chose to bet, not all outcomes, so it is a conditional-on-betting
measure, not a full calibration curve. That is exactly the right lens for "are our bets
making money", but it can't certify the model is calibrated everywhere. The single most
informative number is the model-vs-market Brier comparison on identical bets: if the market's
Brier is lower, the sharp line was closer to reality than our model and we have no edge.

Run: ``python -m dashboard.scorecard``
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from config import configure_logging
from features import confederation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BetResult:
    """One settled YES bet, joined from its order + originating signal."""

    match_id: str
    leg: str  # "H" | "D" | "A"
    team_backed: str | None  # the team a H/A bet backs; None for a draw
    confed: str | None  # that team's confederation, if known
    model_prob: float
    market_implied: float
    edge: float
    staked_cents: int
    pnl_cents: int
    won: bool


@dataclass(frozen=True)
class Summary:
    """Aggregate performance over a set of settled bets."""

    n_bets: int
    n_wins: int
    hit_rate: float
    staked_cents: int
    pnl_cents: int
    roi: float
    avg_model_prob: float
    avg_market_implied: float
    avg_edge: float
    model_brier: float
    market_brier: float


@dataclass(frozen=True)
class CalibrationBucket:
    """A model-probability band with its realized win rate."""

    low: float
    high: float
    n: int
    mean_model_prob: float
    mean_market_implied: float
    win_rate: float


def _decode_leg(match_id: str) -> tuple[str, str | None]:
    """Return ``(leg, team_backed)`` from a ``"{fixture}:{H|D|A}:{home}_{away}"`` id.

    ``team_backed`` is the home team for an H leg, the away team for an A leg, and ``None``
    for a draw (or any match_id that isn't in the expected shape).
    """
    parts = match_id.split(":")
    if len(parts) < 3:
        return "?", None
    leg, teams = parts[1], parts[2].split("_", 1)
    if len(teams) != 2:
        return leg, None
    home, away = teams
    if leg == "H":
        return leg, home
    if leg == "A":
        return leg, away
    return leg, None


def load_settled(conn: sqlite3.Connection) -> list[BetResult]:
    """Load every settled, filled order joined to its signal as a list of ``BetResult``.

    Only rows with a non-null ``pnl_cents`` and a positive filled price count — a settled
    order without a realized P&L or a fill price carries no scoring information.
    """
    rows = conn.execute(
        "SELECT o.contracts, o.filled_price, o.pnl_cents, "
        "s.match_id, s.model_prob, s.market_implied, s.edge "
        "FROM orders o JOIN signals s ON o.signal_id = s.id "
        "WHERE o.status = 'settled' AND o.pnl_cents IS NOT NULL "
        "AND o.filled_price IS NOT NULL "
        "ORDER BY o.settled_at"
    ).fetchall()

    results: list[BetResult] = []
    for row in rows:
        leg, team = _decode_leg(str(row["match_id"]))
        contracts = int(row["contracts"] or 0)
        filled = float(row["filled_price"] or 0.0)
        staked = round(contracts * filled * 100)
        pnl = int(row["pnl_cents"])
        results.append(
            BetResult(
                match_id=str(row["match_id"]),
                leg=leg,
                team_backed=team,
                confed=confederation.confederation_of(team) if team else None,
                model_prob=float(row["model_prob"] or 0.0),
                market_implied=float(row["market_implied"] or 0.0),
                edge=float(row["edge"] or 0.0),
                staked_cents=staked,
                pnl_cents=pnl,
                won=pnl > 0,
            )
        )
    return results


def summarize(results: list[BetResult]) -> Summary:
    """Aggregate hit rate, ROI, and model-vs-market Brier over ``results``.

    Brier is ``mean((prob - outcome)**2)`` with ``outcome`` 1 for a win, 0 for a loss; the
    market Brier uses the line in place of the model probability on the same bets, so a lower
    market Brier means the sharp price was the better estimate and our "edge" was illusory.
    """
    n = len(results)
    if n == 0:
        return Summary(0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wins = sum(1 for r in results if r.won)
    staked = sum(r.staked_cents for r in results)
    pnl = sum(r.pnl_cents for r in results)
    outcomes = [1.0 if r.won else 0.0 for r in results]
    model_brier = sum((r.model_prob - o) ** 2 for r, o in zip(results, outcomes)) / n
    market_brier = (
        sum((r.market_implied - o) ** 2 for r, o in zip(results, outcomes)) / n
    )
    return Summary(
        n_bets=n,
        n_wins=wins,
        hit_rate=wins / n,
        staked_cents=staked,
        pnl_cents=pnl,
        roi=(pnl / staked) if staked else 0.0,
        avg_model_prob=sum(r.model_prob for r in results) / n,
        avg_market_implied=sum(r.market_implied for r in results) / n,
        avg_edge=sum(r.edge for r in results) / n,
        model_brier=model_brier,
        market_brier=market_brier,
    )


def calibration_buckets(
    results: list[BetResult], *, edges: tuple[float, ...] = (0.0, 0.3, 0.4, 0.5, 1.01)
) -> list[CalibrationBucket]:
    """Group bets into model-probability bands and report the realized win rate per band.

    The default bands are coarse on purpose: with a few dozen live bets, finer bins are pure
    noise. A well-calibrated model has ``win_rate`` near ``mean_model_prob`` in each band.
    """
    buckets: list[CalibrationBucket] = []
    for low, high in zip(edges, edges[1:]):
        band = [r for r in results if low <= r.model_prob < high]
        if not band:
            continue
        n = len(band)
        buckets.append(
            CalibrationBucket(
                low=low,
                high=high,
                n=n,
                mean_model_prob=sum(r.model_prob for r in band) / n,
                mean_market_implied=sum(r.market_implied for r in band) / n,
                win_rate=sum(1 for r in band if r.won) / n,
            )
        )
    return buckets


def breakdown(results: list[BetResult], key: str) -> dict[str, Summary]:
    """Per-group summaries keyed by ``leg`` or ``confed`` (groups with no key are skipped)."""
    groups: dict[str, list[BetResult]] = {}
    for r in results:
        value = getattr(r, key)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(r)
    return {name: summarize(rows) for name, rows in groups.items()}


def _verdict(summary: Summary) -> str:
    """One-line read on whether the model is beating the market on real bets."""
    if summary.n_bets < 10:
        return (
            f"[yellow]Only {summary.n_bets} settled bet(s) - too few to conclude. "
            "Keep logging.[/yellow]"
        )
    if summary.model_brier < summary.market_brier and summary.roi > 0:
        return (
            "[green]Model is beating the market on placed bets "
            "(lower Brier, positive ROI).[/green]"
        )
    if summary.model_brier > summary.market_brier:
        return (
            "[red]Market Brier is lower than the model's - the line was the better estimate "
            "on these bets. No demonstrated edge yet.[/red]"
        )
    return "[yellow]Mixed: model Brier is competitive but ROI hasn't confirmed an edge.[/yellow]"


def render(results: list[BetResult], *, console: Console | None = None) -> None:
    """Render the scorecard tables to the console."""
    console = console or Console()
    summary = summarize(results)

    if summary.n_bets == 0:
        console.print("[yellow]No settled bets yet - nothing to score.[/yellow]")
        return

    overview = Table(title="LIVE SCORECARD - SETTLED BETS")
    overview.add_column("Metric")
    overview.add_column("Value", justify="right")
    overview.add_row("Settled bets", str(summary.n_bets))
    overview.add_row("Wins", f"{summary.n_wins} ({summary.hit_rate:.0%})")
    overview.add_row("Staked", f"${summary.staked_cents / 100:,.2f}")
    pnl_color = "green" if summary.pnl_cents >= 0 else "red"
    overview.add_row(
        "P&L", f"[{pnl_color}]${summary.pnl_cents / 100:,.2f}[/{pnl_color}]"
    )
    overview.add_row("ROI", f"[{pnl_color}]{summary.roi:+.1%}[/{pnl_color}]")
    overview.add_row("Avg model prob", f"{summary.avg_model_prob:.1%}")
    overview.add_row("Avg market implied", f"{summary.avg_market_implied:.1%}")
    overview.add_row("Realized win rate", f"{summary.hit_rate:.1%}")
    overview.add_row("Avg edge claimed", f"{summary.avg_edge:+.1%}")
    overview.add_row("Model Brier", f"{summary.model_brier:.3f}")
    overview.add_row("Market Brier", f"{summary.market_brier:.3f}")
    console.print(overview)

    # Avg model prob vs realized win rate is the headline calibration check: if we claim
    # ~40% but win ~20%, the model is overconfident exactly where it cost us.
    console.print(_verdict(summary))

    cal = Table(title="CALIBRATION BY MODEL PROBABILITY")
    for col in ("Band", "N", "Mean model", "Mean market", "Realized"):
        cal.add_column(col, justify="right")
    for b in calibration_buckets(results):
        cal.add_row(
            f"{b.low:.0%}-{min(b.high, 1.0):.0%}",
            str(b.n),
            f"{b.mean_model_prob:.0%}",
            f"{b.mean_market_implied:.0%}",
            f"{b.win_rate:.0%}",
        )
    console.print(cal)

    by_leg = breakdown(results, "leg")
    if by_leg:
        leg_table = Table(title="BY OUTCOME LEG")
        for col in ("Leg", "N", "Hit rate", "ROI"):
            leg_table.add_column(col, justify="right")
        for name in sorted(by_leg):
            s = by_leg[name]
            leg_table.add_row(name, str(s.n_bets), f"{s.hit_rate:.0%}", f"{s.roi:+.0%}")
        console.print(leg_table)

    by_confed = breakdown(results, "confed")
    if by_confed:
        confed_table = Table(title="BY CONFEDERATION BACKED (H/A legs)")
        for col in ("Confed", "N", "Hit rate", "ROI"):
            confed_table.add_column(col, justify="right")
        for name in sorted(by_confed, key=lambda c: by_confed[c].roi):
            s = by_confed[name]
            confed_table.add_row(
                name, str(s.n_bets), f"{s.hit_rate:.0%}", f"{s.roi:+.0%}"
            )
        console.print(confed_table)


def main() -> None:
    configure_logging()
    from data.db import connect, init_db

    init_db()
    with connect() as conn:
        results = load_settled(conn)
    render(results)


if __name__ == "__main__":
    main()
