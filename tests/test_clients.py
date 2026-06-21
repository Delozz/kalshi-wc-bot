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
