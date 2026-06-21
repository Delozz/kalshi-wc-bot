"""Signal generation (strategy/signal_gen.py).

Combines upcoming fixtures, current ELO/form/H2H, the trained model, and live Kalshi
prices into sized, risk-checked signals. It trades all three outcomes (home/draw/away)
as YES contracts, one signal per outcome that clears the edge threshold. The market
resolver (fixture -> per-outcome Kalshi ticker + YES price) is injectable so the
matching can be tested now and refined as the real Kalshi WC market structure is
confirmed.

``generate_signals`` is a pure function over already-fetched inputs (fully testable).
``run_live`` fetches everything (fixtures, markets, history, portfolio) and routes each
signal to the order manager (dry-run by default, L8) and the signals table.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd

from config import settings
from features import elo
from ingestion.api_football import Fixture
from model import predict as predict_mod
from model.dataset import WC_HOSTS, build_live_features
from schemas import Signal
from strategy import edge as edge_mod
from strategy import risk

logger = logging.getLogger(__name__)

# Resolve a fixture to its outcome markets: {"H"|"D"|"A": (ticker, yes_price)}.
MarketResolver = Callable[
    [Fixture, list[dict[str, Any]]], dict[str, "tuple[str, float]"]
]


def _year_of(iso: str) -> int:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).year
    except (ValueError, AttributeError):
        return 2026


def _host_for(
    home: str, away: str, year: int, hosts_by_year: dict[int, set[str]]
) -> str | None:
    hosts = hosts_by_year.get(year, set())
    if home in hosts:
        return home
    if away in hosts:
        return away
    return None


def _open_interest(markets: list[dict[str, Any]], ticker: str) -> float:
    for market in markets:
        if str(market.get("ticker", "")) == ticker:
            # Prefer open_interest_fp (FixedPointCount string); fall back to legacy int.
            val = market.get("open_interest_fp") or market.get("open_interest") or 0
            return float(val)
    return 0.0


def default_outcome_resolver(
    fixture: Fixture, markets: list[dict[str, Any]]
) -> dict[str, tuple[str, float]]:
    """Best-effort: map each outcome (H/D/A) to its Kalshi (ticker, YES price).

    A market belongs to this fixture if its text mentions both teams. The YES outcome is
    read from ``yes_sub_title``: the home team -> H, the away team -> A, "draw"/"tie" ->
    D. Confirm against the real KXWC26 market structure before live trading.
    """
    from ingestion.kalshi import implied_yes_price

    home = fixture.home_team.lower()
    away = fixture.away_team.lower()
    resolved: dict[str, tuple[str, float]] = {}
    for market in markets:
        text = " ".join(
            str(market.get(key, "")) for key in ("title", "subtitle", "ticker")
        ).lower()
        if home not in text or away not in text:
            continue
        price = implied_yes_price(market)
        if price is None:
            continue
        ticker = str(market.get("ticker", ""))
        yes_sub = str(market.get("yes_sub_title", "")).lower()
        if "draw" in yes_sub or "tie" in yes_sub:
            resolved["D"] = (ticker, price)
        elif home in yes_sub:
            resolved["H"] = (ticker, price)
        elif away in yes_sub:
            resolved["A"] = (ticker, price)
    return resolved


def generate_signals(
    *,
    fixtures: list[Fixture],
    history: pd.DataFrame,
    ratings: dict[str, float],
    bundle: dict[str, Any],
    markets: list[dict[str, Any]],
    bankroll_cents: int,
    peak_bankroll_cents: int | None = None,
    open_exposure_cents: int = 0,
    n_open: int = 0,
    resolver: MarketResolver = default_outcome_resolver,
    hosts_by_year: dict[int, set[str]] | None = None,
    threshold: float | None = None,
) -> list[Signal]:
    """Produce sized, risk-checked signals across all outcomes for the given fixtures."""
    hosts_by_year = hosts_by_year or WC_HOSTS
    bankroll = bankroll_cents / 100.0
    peak = (
        peak_bankroll_cents if peak_bankroll_cents is not None else bankroll_cents
    ) / 100.0
    exposure = open_exposure_cents / 100.0

    signals: list[Signal] = []
    for fixture in fixtures:
        outcome_markets = resolver(fixture, markets)
        if not outcome_markets:
            logger.info(
                "No Kalshi markets for %s vs %s", fixture.home_team, fixture.away_team
            )
            continue

        year = _year_of(fixture.kickoff_utc)
        host = _host_for(fixture.home_team, fixture.away_team, year, hosts_by_year)
        features = build_live_features(
            history,
            ratings,
            fixture.home_team,
            fixture.away_team,
            neutral=True,
            host=host,
        )
        probs = predict_mod.predict_outcome(bundle, features)

        for outcome, (ticker, yes_price) in outcome_markets.items():
            model_prob = probs.get(outcome)
            if model_prob is None:
                continue
            signal = edge_mod.build_signal(
                match_id=(
                    f"{fixture.fixture_id}:{outcome}:"
                    f"{fixture.home_team}_{fixture.away_team}"
                ),
                market_ticker=ticker,
                model_prob=model_prob,
                kalshi_yes_price=yes_price,
                bankroll=bankroll,
                threshold=threshold,
            )
            if signal is None:
                continue

            decision = risk.check_all(
                bankroll=bankroll,
                peak_bankroll=peak,
                open_exposure=exposure,
                new_bet=signal["bet_size_cents"] / 100.0,
                n_open=n_open,
                open_interest=_open_interest(markets, ticker),
            )
            if not decision.approved:
                logger.info("Signal %s rejected by risk: %s", ticker, decision.reason)
                continue

            signals.append(signal)
            exposure += signal["bet_size_cents"] / 100.0
            n_open += 1

    logger.info("Generated %d signals from %d fixtures", len(signals), len(fixtures))
    return signals


def _persist(signal: Signal, result: dict[str, Any] | None, *, dry_run: bool) -> None:
    """Persist the signal and, for real (non-dry-run) placements, the order row (L9)."""
    from execution import order_manager

    try:
        from data.db import connect, init_db, log_order, log_signal

        init_db()
        with connect() as conn:
            signal_id = log_signal(conn, signal)
            order_id = result.get("order_id") if result else None
            if not dry_run and order_id and result is not None:
                row = order_manager.build_order_row(
                    order_id=str(order_id),
                    signal_id=signal_id,
                    request=result["request"],
                    status="pending",
                )
                log_order(conn, row)
    except (
        Exception
    ) as exc:  # noqa: BLE001 — persistence must not break signal gen (L9)
        logger.warning(
            "Could not persist signal %s: %s", signal.get("market_ticker"), exc
        )


async def run_live(*, dry_run: bool = True) -> list[Signal]:
    """Fetch live inputs, generate signals, and route them (dry-run by default, L8)."""
    from execution import order_manager, portfolio
    from ingestion import api_football, international_results, kalshi

    bundle = predict_mod.load_bundle()
    if bundle is None:
        return []

    raw_fixtures = await api_football.fetch_fixtures()
    fixtures = api_football.upcoming(api_football.parse_fixtures(raw_fixtures))
    markets = await kalshi.get_markets()
    if not fixtures or not markets:
        logger.warning(
            "No fixtures (%d) or markets (%d); nothing to do",
            len(fixtures),
            len(markets),
        )
        return []

    # Live inference uses ALL known results as feature history. The 2022 holdout rule
    # (L2) governs model DEV/EVAL — using past results as inputs for a 2026 prediction
    # is legitimate, not look-ahead.
    history = international_results.load()
    ratings = elo.final_ratings(history, use_tournament_k=True)

    state = await portfolio.sync_from_kalshi(
        fallback_bankroll_cents=settings.initial_bankroll_cents
    )

    signals = generate_signals(
        fixtures=fixtures,
        history=history,
        ratings=ratings,
        bundle=bundle,
        markets=markets,
        bankroll_cents=state.bankroll_cents,
        peak_bankroll_cents=state.peak_bankroll_cents,
        open_exposure_cents=state.exposure_cents,
        n_open=state.open_count,
    )
    for signal in signals:
        result = await order_manager.place_order(signal, dry_run=dry_run)
        _persist(signal, result, dry_run=dry_run)
    return signals
