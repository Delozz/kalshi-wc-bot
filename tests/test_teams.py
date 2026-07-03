"""Tests for team-name normalization (features/teams.py)."""

from __future__ import annotations

from features.teams import canonical, canonical_market_team


def test_market_team_strips_knockout_decoration() -> None:
    # Knockout KXWCGAME sub-titles decorate the name; the alias must still apply —
    # "Reg Time: USA" silently failed to resolve the whole USA-Belgium fixture.
    assert canonical_market_team("Reg Time: USA") == "United States"
    assert canonical_market_team("Reg Time: Draw") == "Draw"
    assert canonical_market_team("USA") == "United States"  # undecorated unchanged
    assert canonical_market_team("Belgium") == "Belgium"
    assert canonical_market_team("") == ""


def test_known_aliases_map_to_martj42_names() -> None:
    assert canonical("USA") == "United States"
    assert canonical("Korea Republic") == "South Korea"
    assert canonical("IR Iran") == "Iran"
    assert canonical("Czechia") == "Czech Republic"
    # API-Football spelling that cost a fixture on 2026-07-03 (no Kalshi market matched).
    assert canonical("Cape Verde Islands") == "Cape Verde"


def test_unknown_name_passes_through_trimmed() -> None:
    assert canonical("  Brazil  ") == "Brazil"
    assert canonical("France") == "France"


def test_empty_name() -> None:
    assert canonical("") == ""
