"""Tests for API-Football fixture parsing (ingestion/api_football.py), no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ingestion import api_football


def _future_kickoff(hours: float = 48.0) -> str:
    """Return a UTC ISO kickoff string that is always in the future."""
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _raw_fixtures() -> list[dict]:
    return [
        {
            "fixture": {
                "id": 101,
                "date": _future_kickoff(48),
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


def _raw_players() -> list[dict]:
    return [
        {  # two stat blocks -> averaged
            "player": {"id": 1},
            "statistics": [
                {"games": {"rating": "7.0"}},
                {"games": {"rating": "8.0"}},
            ],
        },
        {  # single rating
            "player": {"id": 2},
            "statistics": [{"games": {"rating": "6.5"}}],
        },
        {  # no numeric rating yet -> omitted (degrades to no signal)
            "player": {"id": 3},
            "statistics": [{"games": {"rating": None}}],
        },
        {"player": {}, "statistics": []},  # no id -> skipped
    ]


def test_parse_player_ratings_averages_and_skips_unrated() -> None:
    ratings = api_football._parse_player_ratings(_raw_players())
    assert ratings == {1: 7.5, 2: 6.5}
    assert 3 not in ratings  # unrated players never fabricate strength
