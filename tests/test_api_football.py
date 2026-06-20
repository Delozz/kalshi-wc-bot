"""Tests for API-Football fixture parsing (ingestion/api_football.py), no network."""

from __future__ import annotations

from ingestion import api_football


def _raw_fixtures() -> list[dict]:
    return [
        {
            "fixture": {
                "id": 101,
                "date": "2026-06-20T18:00:00+00:00",
                "status": {"short": "NS"},
            },
            "teams": {"home": {"name": "USA"}, "away": {"name": "Wales"}},
            "league": {"round": "Group Stage - 1"},
        },
        {
            "fixture": {
                "id": 102,
                "date": "2026-06-15T15:00:00+00:00",
                "status": {"short": "FT"},
            },
            "teams": {"home": {"name": "Brazil"}, "away": {"name": "Serbia"}},
            "league": {"round": "Group Stage - 1"},
        },
        {"fixture": {"id": 103}},  # malformed — missing teams
    ]


def test_parse_fixtures_normalizes_names_and_skips_bad_rows() -> None:
    fixtures = api_football.parse_fixtures(_raw_fixtures())
    assert len(fixtures) == 2
    usa = fixtures[0]
    assert usa.home_team == "United States"  # canonicalized from "USA"
    assert usa.away_team == "Wales"
    assert usa.fixture_id == 101
    assert usa.status == "NS"


def test_upcoming_filters_to_prematch() -> None:
    fixtures = api_football.parse_fixtures(_raw_fixtures())
    upcoming = api_football.upcoming(fixtures)
    assert len(upcoming) == 1
    assert upcoming[0].fixture_id == 101
