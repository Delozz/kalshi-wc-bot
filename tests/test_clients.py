"""Tests for the API-client pure helpers (no network): price + no-vig parsing."""

from __future__ import annotations

import math

from ingestion import kalshi, odds_api


def test_kalshi_base_url_demo_by_default() -> None:
    # .env is absent in CI, so KALSHI_ENV defaults to demo (L8 — safe default).
    assert kalshi.base_url().startswith((kalshi.DEMO_HOST, kalshi.PROD_HOST))
    assert kalshi.base_url().endswith(kalshi.PATH_PREFIX)


def test_implied_yes_price_from_ask() -> None:
    # Current API: yes_ask_dollars is a FixedPointDollars string already in 0–1 range.
    assert kalshi.implied_yes_price({"yes_ask_dollars": "0.62"}) == 0.62
    # Legacy fallback: yes_ask was an integer in cents.
    assert kalshi.implied_yes_price({"yes_ask": 62}) == 0.62
    assert kalshi.implied_yes_price({"yes_bid": 60}) is None


def test_novig_from_h2h_sums_to_one() -> None:
    outcomes = [
        {"name": "France", "price": 1.95},
        {"name": "Draw", "price": 3.5},
        {"name": "Morocco", "price": 4.2},
    ]
    fair = odds_api.novig_from_h2h(outcomes)
    assert math.isclose(sum(fair.values()), 1.0)
    assert fair["France"] > fair["Morocco"]  # shorter odds -> higher probability


def test_novig_from_h2h_rejects_bad_odds() -> None:
    assert odds_api.novig_from_h2h([{"name": "X", "price": 0.0}]) == {}
    assert odds_api.novig_from_h2h([{"name": "X"}]) == {}


def _book(
    home: str, away: str, h: float, d: float, a: float, key: str = "book"
) -> dict:
    """One bookmaker entry in the Odds API event shape (decimal h2h odds)."""
    return {
        "key": key,
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": home, "price": h},
                    {"name": "Draw", "price": d},
                    {"name": away, "price": a},
                ],
            }
        ],
    }


def test_consensus_book_probs_median_across_books() -> None:
    event = {
        "home_team": "France",
        "away_team": "Morocco",
        "bookmakers": [
            _book("France", "Morocco", 2.0, 3.5, 4.0, key="a"),
            _book("France", "Morocco", 2.1, 3.5, 3.8, key="b"),
            _book("France", "Morocco", 9.0, 3.5, 1.3, key="c"),  # stale outlier line
        ],
    }
    consensus = odds_api.consensus_book_probs([event])
    probs = consensus[frozenset({"France", "Morocco"})]
    assert math.isclose(sum(probs.values()), 1.0)
    # The median tracks the two agreeing books, not the outlier: France stays the favorite.
    assert probs["France"] > probs["Morocco"]
    assert probs["France"] > 0.4


def test_consensus_book_probs_canonicalizes_names() -> None:
    # Odds API spellings differ from martj42; the consensus must key by canonical names so
    # signal_gen's fixture lookup (already canonical) matches.
    event = {
        "home_team": "Korea Republic",
        "away_team": "USA",
        "bookmakers": [_book("Korea Republic", "USA", 2.5, 3.2, 2.9)],
    }
    consensus = odds_api.consensus_book_probs([event])
    probs = consensus[frozenset({"South Korea", "United States"})]
    assert set(probs) == {"South Korea", "United States", "Draw"}


def test_consensus_book_probs_skips_unusable_events() -> None:
    incomplete = {
        "home_team": "France",
        "away_team": "Morocco",
        "bookmakers": [
            {
                "key": "x",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "France", "price": 2.0},
                            {"name": "Morocco", "price": 3.0},
                        ],
                    }
                ],
            }  # two-way quote (no Draw): can't be oriented onto a 3-way market
        ],
    }
    no_h2h = {
        "home_team": "Spain",
        "away_team": "Chile",
        "bookmakers": [{"key": "y", "markets": [{"key": "totals", "outcomes": []}]}],
    }
    assert odds_api.consensus_book_probs([incomplete, no_h2h, {}]) == {}
    assert odds_api.consensus_book_probs([]) == {}
