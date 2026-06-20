"""Signal generation (strategy/signal_gen.py).

Combines upcoming fixtures, current ELO/form/H2H, the trained model, and live Kalshi
prices into sized, risk-checked signals. v1 trades only the home-win YES contract; the
market resolver (fixture -> Kalshi ticker + YES price) is injectable so the matching can
be tested now and refined as the real Kalshi WC market structure is confirmed.

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

MarketResolver = Callable[[Fixture, list[dict[str, Any]]], "tuple[str, float] | None"]


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
            return float(market.get("open_interest", 0) or 0)
    return 0.0


def default_market_resolver(
    fixture: Fixture, markets: list[dict[str, Any]]
) -> tuple[str, float] | None:
    """Best-effort match: a market whose text mentions both teams; YES = home win."""
    from ingestion.kalshi import implied_yes_price

    home = fixture.home_team.lower()
    away = fixture.away_team.lower()
    for market in markets:
        text = " ".join(
            str(market.get(key, ""))
            for key in ("title", "subtitle", "yes_sub_title", "ticker")
        ).lower()
        if home in text and away in text:
            price = implied_yes_price(market)
            if price is not None:
                return (str(market.get("ticker", "")), price)
    return None


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
    resolver: MarketResolver = default_market_resolver,
    hosts_by_year: dict[int, set[str]] | None = None,
    threshold: float | None = None,
) -> list[Signal]:
    """Produce sized, risk-checked home-win signals for the given fixtures."""
    hosts_by_year = hosts_by_year or WC_HOSTS
    bankroll = bankroll_cents / 100.0
    peak = (
        peak_bankroll_cents if peak_bankroll_cents is not None else bankroll_cents
    ) / 100.0
    exposure = open_exposure_cents / 100.0

    signals: list[Signal] = []
    for fixture in fixtures:
        resolved = resolver(fixture, markets)
        if resolved is None:
            logger.info(
                "No Kalshi market for %s vs %s", fixture.home_team, fixture.away_team
            )
            continue
        ticker, yes_price_home = resolved

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

        signal = edge_mod.build_signal(
            match_id=f"{fixture.home_team}_{fixture.away_team}_{fixture.fixture_id}",
            market_ticker=ticker,
            model_prob=probs["H"],
            kalshi_yes_price=yes_price_home,
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


def _persist_signal(signal: Signal) -> None:
    try:
        from data.db import connect, init_db, log_signal

        init_db()
        with connect() as conn:
            log_signal(conn, signal)
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
        await order_manager.place_order(signal, dry_run=dry_run)
        _persist_signal(signal)
    return signals
