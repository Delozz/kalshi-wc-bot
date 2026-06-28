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
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from config import settings
from features import confederation, elo, lineup, squad
from ingestion.api_football import Fixture
from model import dixon_coles
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

# A probability engine: maps a fixture (+ its prebuilt feature dict) to {H, D, A} probs.
# This is the one seam between signal generation and the model, so the engine is
# swappable (classifier vs Dixon-Coles) without touching edge/risk/sizing downstream.
Predictor = Callable[[Fixture, dict[str, float]], dict[str, float]]


def _classifier_predictor(bundle: dict[str, Any]) -> Predictor:
    """The logistic/XGBoost bundle engine: predicts from the engineered feature vector."""

    def predict(_fixture: Fixture, features: dict[str, float]) -> dict[str, float]:
        return predict_mod.predict_outcome(bundle, features)

    return predict


def _dc_predictor(model: dixon_coles.DixonColesModel) -> Predictor:
    """The Dixon-Coles goals engine: predicts from team names off the fitted score model.

    WC venues are neutral, so home advantage stays off — consistent with the feature path,
    which also builds features with ``neutral=True``. The feature dict is ignored (DC reads
    team strength from its own fitted attack/defense ratings)."""

    def predict(fixture: Fixture, _features: dict[str, float]) -> dict[str, float]:
        return model.predict_hda(fixture.home_team, fixture.away_team, neutral=True)

    return predict


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


def _event_prefix(ticker: str) -> str:
    """The event identity shared by a fixture's H/D/A legs (ticker minus the outcome suffix).

    KXWCGAME structures one event per match with three per-outcome tickers that differ only
    in their final segment, e.g. ``KXWCGAME-26JUN27CODUZB-TIE`` / ``-UZB`` / ``-COD`` all
    share ``KXWCGAME-26JUN27CODUZB``. Stripping the last ``-`` segment yields that shared
    key, which is how we enforce one bet per match (and detect a fixture we already hold).
    """
    return ticker.rsplit("-", 1)[0]


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
    """Map each outcome (H/D/A) to its Kalshi (ticker, YES price).

    Groups markets by their shared event prefix (the ticker portion before the final '-'),
    which is how KXWCGAME structures its 3-leg markets.  A group belongs to this fixture
    when both team names appear in the ``yes_sub_title`` fields of the group's markets.
    Falls back to deprecated ``title``/``subtitle`` text search for other market formats.
    """
    from features.teams import canonical
    from ingestion.kalshi import implied_yes_price

    home = fixture.home_team.lower()
    away = fixture.away_team.lower()
    resolved: dict[str, tuple[str, float]] = {}

    # Group by event_ticker (explicit) or by ticker prefix (KXWCGAME convention).
    groups: dict[str, list[dict[str, Any]]] = {}
    for market in markets:
        key = str(
            market.get("event_ticker") or market.get("ticker", "").rsplit("-", 1)[0]
        )
        groups.setdefault(key, []).append(market)

    for _key, group in groups.items():
        # Apply canonical() so "Turkiye"→"Turkey", "Congo DR"→"DR Congo", etc.
        # Fixture names are already canonical; yes_sub_title values are raw Kalshi strings.
        canonical_subs = [
            canonical(str(m.get("yes_sub_title", ""))).lower() for m in group
        ]
        has_home = any(home in s for s in canonical_subs)
        has_away = any(away in s for s in canonical_subs)

        if not (has_home and has_away):
            # Fallback: deprecated title/subtitle text for non-KXWCGAME formats.
            text = " ".join(
                str(m.get(k, "")) for m in group for k in ("title", "subtitle")
            ).lower()
            has_home = home in text
            has_away = away in text

        if not (has_home and has_away):
            continue

        for market in group:
            price = implied_yes_price(market)
            if price is None:
                continue
            ticker = str(market.get("ticker", ""))
            yes_sub = canonical(str(market.get("yes_sub_title", ""))).lower()
            if "draw" in yes_sub or "tie" in yes_sub:
                resolved["D"] = (ticker, price)
            elif home in yes_sub:
                resolved["H"] = (ticker, price)
            elif away in yes_sub:
                resolved["A"] = (ticker, price)

        if resolved:
            break  # found the matching event group

    return resolved


def generate_signals(
    *,
    fixtures: list[Fixture],
    history: pd.DataFrame,
    ratings: dict[str, float],
    bundle: dict[str, Any] | None = None,
    markets: list[dict[str, Any]],
    bankroll_cents: int,
    peak_bankroll_cents: int | None = None,
    open_exposure_cents: int = 0,
    n_open: int = 0,
    held_tickers: set[str] | None = None,
    resolver: MarketResolver = default_outcome_resolver,
    hosts_by_year: dict[int, set[str]] | None = None,
    threshold: float | None = None,
    lineups_by_fixture: (
        dict[int, tuple[dict[str, Any] | None, dict[str, Any] | None]] | None
    ) = None,
    squad_ratings_by_team: dict[str, dict[int, float]] | None = None,
    predictor: Predictor | None = None,
) -> list[Signal]:
    """Produce sized, risk-checked signals across all outcomes for the given fixtures.

    ``lineups_by_fixture`` (optional) maps a fixture id to its ``(home_lineup,
    away_lineup)`` API-Football payloads. When present, the resulting home-minus-away
    strength delta nudges the model probability per outcome (positive favours home, so
    it is negated for the away leg and zeroed for the draw). Absent or unannounced
    lineups leave a 0.0 delta — identical to the pre-lineup behaviour.

    ``squad_ratings_by_team`` (optional) maps a canonical team name to its squad's season
    player ratings ``{player_id: rating}``. When present, an always-on squad-strength prior
    tilts the full {H, D, A} vector toward the stronger squad *before* the edge check —
    available at any horizon, so it shapes advance bets where the announced-lineup nudge is
    still 0.0. Teams with no ratings contribute a 0.0 delta (zero-impact fallback).

    ``predictor`` (optional) is the probability engine — a ``(fixture, features) -> {H,D,A}``
    callable. Defaults to the classifier ``bundle``; ``run_live`` injects the Dixon-Coles
    engine instead when ``MODEL_ENGINE=dc``. The squad nudge and risk filters run identically
    on whatever base probabilities the engine returns.

    ``held_tickers`` (optional) is the set of market tickers we already hold an open position
    in. Any candidate on one of them is skipped (no averaging into a position), and — because
    only one bet is taken per fixture — holding any leg of a match also blocks its sibling
    legs across cycles, so the bot never ends up with two correlated legs on one game.
    ``run_live`` passes the live portfolio's tickers; the position/exposure caps still apply
    on top.

    Only the single highest-edge leg of any fixture is bet: the H/D/A legs of a 3-way market
    are mutually exclusive, so stacking them concentrates the stake on one match and the
    draw+underdog pair is one "favorite doesn't win" bet placed twice."""
    hosts_by_year = hosts_by_year or WC_HOSTS
    # Markets we already hold a position in — never re-bet them (no averaging into an open
    # position across cycles), so each market gets at most one entry.
    held = held_tickers or set()
    # Engine seam: use the injected predictor, else fall back to the classifier bundle.
    predict = (
        predictor if predictor is not None else _classifier_predictor(bundle or {})
    )
    bankroll = bankroll_cents / 100.0
    peak = (
        peak_bankroll_cents if peak_bankroll_cents is not None else bankroll_cents
    ) / 100.0
    exposure = open_exposure_cents / 100.0

    # Phase 1: build every candidate signal that clears the edge threshold, across all
    # fixtures/outcomes, WITHOUT applying the sequential position/exposure caps yet. We
    # tag each with its ticker so the liquidity check can run in phase 2.
    candidates: list[tuple[Signal, str]] = []
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
        probs = predict(fixture, features)

        # Confederation-drift correction (engine-agnostic, before any finer prior): raw ELO
        # over-rates AFC/CONCACAF and under-rates UEFA/CONMEBOL because each pool is only
        # loosely anchored to the others, which manufactured the Japan-over-Brazil / Iran /
        # Australia overprices. Tilt the {H,D,A} vector by the home-minus-away confederation
        # ELO offset. Zero for intra-confederation games (offsets cancel) or unmapped teams.
        conf_elo_delta = confederation.elo_delta(fixture.home_team, fixture.away_team)
        if conf_elo_delta:
            probs = edge_mod.apply_confederation_prior(probs, conf_elo_delta)

        # Always-on squad-strength prior: tilt the {H,D,A} vector toward the stronger
        # squad. Squad ratings are available at any horizon (unlike announced lineups), so
        # this shapes advance bets — the Portugal/Ronaldo, Norway/Haaland case where we bet
        # days before kickoff and the lineup nudge would still be 0.0.
        squad_delta = 0.0
        if squad_ratings_by_team:
            squad_delta = squad.squad_strength_delta(
                squad_ratings_by_team.get(fixture.home_team),
                squad_ratings_by_team.get(fixture.away_team),
            )
            if squad_delta:
                probs = edge_mod.apply_squad_prior(probs, squad_delta)

        # Pre-edge signal-quality gate inputs: the ELO favorite and the size of its edge.
        # Draw/underdog legs against an overwhelming favorite are where our model is least
        # trustworthy and the Kalshi line sharpest (the Iraq-over-France problem). Fold the
        # confederation offset into the ELOs here too, so the favorite and the powerhouse-gap
        # reflect true cross-confederation strength (e.g. Paraguay, not Australia, is the
        # favorite once the offsets are applied) and the ratio/powerhouse guards key off it.
        conf_w = settings.confederation_weight
        home_elo = features["home_elo_pre"] + conf_w * confederation.offset_for(
            fixture.home_team
        )
        away_elo = features["away_elo_pre"] + conf_w * confederation.offset_for(
            fixture.away_team
        )
        favorite = "H" if home_elo >= away_elo else "A"
        elo_gap = abs(home_elo - away_elo)
        # The squad prior independently confirms the ELO favorite when its sign agrees,
        # which lowers the ELO bar for suppressing draw/upset legs (the Portugal case).
        squad_confirms_favorite = (squad_delta > 0 and favorite == "H") or (
            squad_delta < 0 and favorite == "A"
        )

        # Home-minus-away starting-XI strength (0.0 when lineups aren't announced).
        raw_lineup_delta = 0.0
        if lineups_by_fixture:
            pair = lineups_by_fixture.get(fixture.fixture_id)
            if pair is not None:
                raw_lineup_delta = lineup.lineup_strength_delta(pair[0], pair[1])

        for outcome, (ticker, yes_price) in outcome_markets.items():
            model_prob = probs.get(outcome)
            if model_prob is None:
                continue
            # Never add to a market we already hold: skip it entirely so a persistent edge
            # can't grow the same position across cycles (the topping-up we want to avoid).
            if ticker in held:
                logger.info(
                    "Skipping %s %s leg: already holding this market (no re-bet)",
                    ticker,
                    outcome,
                )
                continue
            # Drop legs the market is better-informed on (longshot floor, model/market
            # mismatch, draw/upset vs an overwhelming favorite) before edge/sizing.
            admit = risk.outcome_admissible(
                bet_on_favorite=(outcome == favorite),
                model_prob=model_prob,
                market_price=yes_price,
                favorite_elo_gap=elo_gap,
                squad_confirms_favorite=squad_confirms_favorite,
            )
            if not admit.approved:
                logger.info(
                    "Filtered %s %s leg: %s (model=%.3f, price=%.2f, elo_gap=%.0f)",
                    ticker,
                    outcome,
                    admit.reason,
                    model_prob,
                    yes_price,
                    elo_gap,
                )
                continue
            # Sign the delta for this leg: +home, -away, 0 for the (ambiguous) draw.
            if outcome == "H":
                signed_delta = raw_lineup_delta
            elif outcome == "A":
                signed_delta = -raw_lineup_delta
            else:
                signed_delta = 0.0
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
                lineup_delta=signed_delta,
            )
            if signal is None:
                continue
            candidates.append((signal, ticker))

    # Phase 2: rank candidates by edge (best first) so the strongest bets claim the
    # scarce position/exposure slots — not whichever fixture happened to be parsed first
    # (previously this followed API-Football's chronological order, starving high-edge
    # later matches once max_positions filled).
    candidates.sort(key=lambda c: c[0]["edge"], reverse=True)

    # One bet per fixture. Every YES leg on a 3-way match is mutually exclusive — at most
    # one can win — so stacking legs concentrates the whole stake on a single game, and the
    # draw+underdog pair is really one "favorite doesn't win" bet placed twice (both lost
    # together on DR Congo, France, Portugal, …). Because candidates are edge-sorted, taking
    # only the first leg per event keeps the single highest-edge expression of each match.
    # Seed with the events we already hold so a sibling leg can't be added across cycles.
    claimed_events: set[str] = {_event_prefix(t) for t in held}

    signals: list[Signal] = []
    for signal, ticker in candidates:
        event = _event_prefix(ticker)
        if event in claimed_events:
            logger.info(
                "Signal %s skipped: already betting this fixture (one bet per match)",
                ticker,
            )
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
        claimed_events.add(event)
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


async def _attach_player_ratings(entry: dict[str, Any]) -> None:
    """Inject each starting player's season rating into a lineup entry, in place.

    The lineup endpoint returns the XI but no strength signal, so we fold in season
    ratings from ``/players``. Players the rating source doesn't cover keep no ``rating``
    key and are simply skipped by ``lineup_strength_delta``. Any failure is swallowed so
    one team never blocks the run (L9)."""
    from ingestion import api_football

    team_id = (entry.get("team") or {}).get("id")
    if team_id is None:
        return
    try:
        ratings = await api_football.fetch_squad_ratings(int(team_id))
    except Exception as exc:  # noqa: BLE001 — L9: a rating gap must not crash the run
        logger.warning("Squad-rating fetch failed for team %s: %s", team_id, exc)
        return
    for slot in entry.get("startXI") or []:
        player = (slot or {}).get("player") or {}
        pid = player.get("id")
        if pid is not None and int(pid) in ratings:
            player["rating"] = ratings[int(pid)]


async def _fetch_lineups(
    fixtures: list[Fixture],
) -> dict[int, tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """Fetch announced lineups for fixtures kicking off within the next 3 hours.

    Lineups are typically released ~1h before kickoff, so we only spend API calls inside
    a tight pre-match window; matches further out (or already started) are skipped and
    simply carry no lineup signal. Each fixture maps to ``(home_lineup, away_lineup)``,
    matched by canonical team name. Any fetch failure is swallowed so one bad fixture
    never blocks the rest (L9)."""
    from features.teams import canonical
    from ingestion import api_football

    now = datetime.now(timezone.utc)
    out: dict[int, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
    for fixture in fixtures:
        try:
            kickoff = datetime.fromisoformat(fixture.kickoff_utc.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if not (now <= kickoff <= now + timedelta(hours=3)):
            continue
        try:
            raw = await api_football.fetch_lineups(fixture.fixture_id)
        except Exception as exc:  # noqa: BLE001 — L9: never let one fixture crash run
            logger.warning("Lineup fetch failed for %s: %s", fixture.fixture_id, exc)
            continue
        if not raw:
            continue
        home_lu: dict[str, Any] | None = None
        away_lu: dict[str, Any] | None = None
        for entry in raw:
            await _attach_player_ratings(entry)
            name = canonical(str((entry.get("team") or {}).get("name", "")))
            if name == fixture.home_team:
                home_lu = entry
            elif name == fixture.away_team:
                away_lu = entry
        if home_lu or away_lu:
            out[fixture.fixture_id] = (home_lu, away_lu)
    if out:
        logger.info("Fetched announced lineups for %d fixture(s)", len(out))
    return out


async def _fetch_squad_ratings(
    raw_fixtures: list[dict[str, Any]], fixtures: list[Fixture]
) -> dict[str, dict[int, float]]:
    """Fetch cached squad season ratings for every upcoming team we have an API id for.

    Unlike lineups, squad ratings exist at any horizon, so this runs for every upcoming
    fixture (not just the pre-match window). Team ids are read from the raw API-Football
    fixtures payload; the Kalshi-fallback path carries no ids, so this simply returns ``{}``
    — a zero-impact prior. Each fetch is cached per team per day and wrapped so one bad team
    never blocks the rest (L9)."""
    from features.teams import canonical
    from ingestion import api_football

    name_to_id: dict[str, int] = {}
    for fixture in raw_fixtures or []:
        teams = fixture.get("teams") or {}
        for side in ("home", "away"):
            team = teams.get(side) or {}
            tid, name = team.get("id"), team.get("name")
            if tid is not None and name:
                name_to_id[canonical(str(name))] = int(tid)

    wanted = {f.home_team for f in fixtures} | {f.away_team for f in fixtures}
    out: dict[str, dict[int, float]] = {}
    for name in wanted:
        team_id = name_to_id.get(name)
        if team_id is None:
            continue
        try:
            ratings = await api_football.fetch_squad_ratings(team_id)
        except (
            Exception
        ) as exc:  # noqa: BLE001 — L9: a rating gap must not crash the run
            logger.warning("Squad-rating fetch failed for %s: %s", name, exc)
            continue
        if ratings:
            out[name] = ratings
    if out:
        logger.info("Fetched squad ratings for %d team(s)", len(out))
    return out


async def run_live(*, dry_run: bool = True) -> list[Signal]:
    """Fetch live inputs, generate signals, and route them (dry-run by default, L8)."""
    from execution import order_manager, portfolio
    from ingestion import api_football, international_results, kalshi

    raw_fixtures = await api_football.fetch_fixtures()
    fixtures = api_football.upcoming(api_football.parse_fixtures(raw_fixtures))
    # Lineups are only fetchable for real API-Football fixture ids; the Kalshi fallback
    # below synthesises ids, so lineup enrichment is skipped on that path.
    fixtures_have_real_ids = bool(fixtures)
    markets = await kalshi.get_markets(status="open")
    if not markets:
        logger.warning("No Kalshi markets; nothing to do")
        return []
    if not fixtures:
        # API-Football free tier blocks 2026 season — derive fixtures from KXWCGAME
        # markets instead (team names live in yes_sub_title; dates in ticker).
        fixtures = kalshi.parse_wc_fixtures(markets)
    if not fixtures:
        logger.warning("No upcoming WC fixtures from any source; nothing to do")
        return []

    # Live inference uses ALL known results as feature history. The 2022 holdout rule
    # (L2) governs model DEV/EVAL — using past results as inputs for a 2026 prediction
    # is legitimate, not look-ahead.
    history = await international_results.load_async()
    ratings = elo.final_ratings(history, use_tournament_k=True)

    # Always-on squad-strength prior — runs at any horizon, so it shapes advance bets the
    # lineup nudge can't reach. Empty (zero-impact) on the Kalshi-fallback path (no ids).
    # Fetched here (before the DC fit) so its top-11 strength can also seed the goals model.
    squad_ratings_by_team = await _fetch_squad_ratings(raw_fixtures, fixtures)

    # Build the live probability engine (config-selectable; DC is the validated default,
    # MODEL_ENGINE=classifier rolls back instantly). DC is fit once per run on the full
    # history with ELO as the primary strength prior at the validated scale, and squad
    # strength blended in as a secondary prior so star-laden squads lift their own
    # attack/defense at fit time (refines data-sparse teams; data-rich teams override it).
    if settings.model_engine == "dc":
        squad_strength_by_team = {
            team: strength
            for team, team_ratings in squad_ratings_by_team.items()
            if (strength := squad.squad_strength(team_ratings)) is not None
        }
        dc_model = dixon_coles.fit(
            history,
            cutoff=pd.Timestamp(datetime.now(timezone.utc)),
            strength=ratings,
            prior_scale=dixon_coles.DEFAULT_PRIOR_SCALE,
            secondary=squad_strength_by_team or None,
            secondary_weight=settings.dc_squad_prior_weight,
            # Drop teams with too little history from the goals fit: with only a handful of
            # matches the MLE attack/defense is noise (and over-rates minnows who ran up
            # scores on regional weaklings). Every real WC side has 350+ internationals, so
            # 12 only sheds unofficial/representative teams (Yorkshire, Kabylia, …) that
            # pollute the league-average baseline. Dropped teams fall back to the ELO prior.
            min_matches=12,
        )
        predictor = _dc_predictor(dc_model)
        logger.info(
            "Live engine: Dixon-Coles (ELO prior scale=%.2f, squad prior weight=%.2f "
            "over %d team(s))",
            dixon_coles.DEFAULT_PRIOR_SCALE,
            settings.dc_squad_prior_weight,
            len(squad_strength_by_team),
        )
    else:
        bundle = predict_mod.load_bundle()
        if bundle is None:
            return []
        predictor = _classifier_predictor(bundle)
        logger.info("Live engine: classifier bundle")

    state = await portfolio.sync_from_kalshi(
        fallback_bankroll_cents=settings.initial_bankroll_cents
    )
    # Markets we already hold live on Kalshi — the primary no-re-bet set.
    held_tickers: set[str] = {p.ticker for p in state.positions}
    # Lift peak to the real high-water mark from the ledger so the stop-loss measures a
    # genuine drawdown (sync alone sets peak == current balance), and union in every ticker
    # we have ever ordered so a settled/liquidated market can't be re-entered (the Kalshi
    # positions endpoint forgets closed positions). L9: never crash on this.
    try:
        from data.db import connect, ordered_tickers, real_peak_bankroll

        with connect() as conn:
            portfolio.ratchet_peak(state, real_peak_bankroll(conn))
            held_tickers |= ordered_tickers(conn)
    except (
        Exception
    ) as exc:  # noqa: BLE001 — peak read must not block signal generation
        logger.warning("Could not read historical peak/orders: %s", exc)

    lineups_by_fixture = (
        await _fetch_lineups(fixtures) if fixtures_have_real_ids else {}
    )

    signals = generate_signals(
        fixtures=fixtures,
        history=history,
        ratings=ratings,
        predictor=predictor,
        markets=markets,
        bankroll_cents=state.bankroll_cents,
        peak_bankroll_cents=state.peak_bankroll_cents,
        open_exposure_cents=state.exposure_cents,
        n_open=state.open_count,
        held_tickers=held_tickers,
        lineups_by_fixture=lineups_by_fixture,
        squad_ratings_by_team=squad_ratings_by_team,
    )
    for signal in signals:
        result = await order_manager.place_order(signal, dry_run=dry_run)
        _persist(signal, result, dry_run=dry_run)
        if not dry_run and result and result.get("order_id"):
            await _finalize_fill(str(result["order_id"]), result["request"])
    return signals


async def _finalize_fill(order_id: str, request: Any) -> None:
    """Poll a live order to its fill and record the price/status (closes the place->fill
    gap that previously left orders stuck at 'pending', unsettleable). Any failure is
    swallowed so one order never blocks the rest (L9); the order simply stays 'pending'
    and the next settle/sync pass can reconcile it."""
    from execution import order_manager

    try:
        status, fill_cents = await order_manager.confirm_fill(
            order_id, fallback_price_cents=request.limit_price_cents
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 — L9: a fill-poll failure must not crash run
        logger.warning("Fill confirmation failed for %s: %s", order_id, exc)
        return

    try:
        from data.db import connect, update_order_status

        with connect() as conn:
            update_order_status(
                conn,
                order_id,
                status,
                filled_price=(fill_cents / 100.0 if fill_cents is not None else None),
            )
    except Exception as exc:  # noqa: BLE001 — L9: persistence must not crash run
        logger.warning("Could not record fill for %s: %s", order_id, exc)
        return
    logger.info("Order %s resolved: %s (fill=%s c)", order_id, status, fill_cents)
