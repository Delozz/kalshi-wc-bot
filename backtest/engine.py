"""Backtest engine (backtest/engine.py) — time-ordered event loop.

Phase 1 validation: a flat-bet baseline that buys YES on the home team at the
Pinnacle no-vig implied price for every match in the selected year, settling against
the real result. It demonstrates the full ingest -> features -> simulate -> settle ->
metrics loop with zero look-ahead (every feature build routes through the guard, L1).

Data note: football-data.co.uk serves domestic-league odds CSVs, not a
WC-with-odds file, so this baseline runs on real fetched league data as the Phase 1
substrate. National-team World Cup data swaps in once an odds source is confirmed.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from backtest import metrics as metrics_mod
from backtest import simulator
from config import configure_logging, settings
from features import pipeline as feature_pipeline
from ingestion import football_data_co

logger = logging.getLogger(__name__)

# Demo substrate: league-seasons overlapping each tournament year. The 2022 entry is
# ordinary league data for that calendar year — NOT the StatsBomb WC holdout (L2).
_SEASON_SETS: dict[int, list[tuple[str, str]]] = {
    2018: [("E0", "1718"), ("E0", "1819")],
    2022: [("E0", "2122"), ("E0", "2223")],
}


def run_backtest(
    year: int, *, initial_bankroll: float | None = None
) -> metrics_mod.BacktestMetrics:
    """Run the flat-bet baseline backtest for a calendar year."""
    bankroll = (
        initial_bankroll
        if initial_bankroll is not None
        else settings.initial_bankroll_cents / 100.0
    )
    pairs = _SEASON_SETS.get(year)
    if not pairs:
        raise ValueError(f"No season set configured for year {year}")

    paths = football_data_co.download_seasons_sync(pairs)
    matches = football_data_co.load_matches(paths)
    if matches.empty:
        raise SystemExit("No matches loaded; aborting backtest")

    universe = matches[
        (matches["date"].dt.year == year)
        & matches["psh"].notna()
        & matches["ftr"].notna()
    ].sort_values("date")
    logger.info(
        "Loaded %d matches; %d in the %d bet universe",
        len(matches),
        len(universe),
        year,
    )

    equity: list[float] = [bankroll]
    pnls: list[float] = []
    wins: list[bool] = []
    returns: list[float] = []

    for row in universe.itertuples(index=False):
        # Pass only causal history (strictly before kickoff); the guard then runs as a
        # hard tripwire inside build_match_features. Day-granular dates mean same-day
        # matches are conservatively excluded — no intraday leak.
        past = matches[matches["date"] < row.date]
        feats = feature_pipeline.build_match_features(
            past,
            home_team=row.home_team,
            away_team=row.away_team,
            cutoff=row.date.to_pydatetime(),
            match_odds={"psh": row.psh, "psd": row.psd, "psa": row.psa},
        )
        price = feats["pinnacle_implied_home"]
        if pd.isna(price):
            continue
        won = bool(row.ftr == "H")
        fill = simulator.simulate_yes_fill(price, won)
        bankroll += fill.pnl
        equity.append(bankroll)
        pnls.append(fill.pnl)
        wins.append(won)
        returns.append(fill.pnl / fill.price if fill.price > 0 else 0.0)

    result = metrics_mod.compute_metrics(returns, pnls, wins, equity, equity[0])
    logger.info(
        "Backtest %d | bets=%d hit=%.1f%% ROI=%.2f%% PnL=$%.2f Sharpe=%.2f "
        "MaxDD=%.1f%% final=$%.2f",
        year,
        result.n_bets,
        result.hit_rate * 100.0,
        result.roi * 100.0,
        result.total_pnl,
        result.sharpe,
        result.max_drawdown * 100.0,
        result.final_bankroll,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a flat-bet baseline backtest.")
    parser.add_argument(
        "--tournament",
        type=int,
        default=2018,
        help="Calendar year to backtest (2018 dev).",
    )
    args = parser.parse_args()
    configure_logging()
    if args.tournament == 2022:
        logger.warning(
            "2022 is the model holdout year (L2). This uses ordinary league demo "
            "data, not the StatsBomb WC holdout set."
        )
    run_backtest(args.tournament)


if __name__ == "__main__":
    main()
