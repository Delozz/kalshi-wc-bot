"""Tests for team-name normalization (features/teams.py)."""

from __future__ import annotations

from features.teams import canonical


def test_known_aliases_map_to_martj42_names() -> None:
    assert canonical("USA") == "United States"
    assert canonical("Korea Republic") == "South Korea"
    assert canonical("IR Iran") == "Iran"
    assert canonical("Czechia") == "Czech Republic"


def test_unknown_name_passes_through_trimmed() -> None:
    assert canonical("  Brazil  ") == "Brazil"
    assert canonical("France") == "France"


def test_empty_name() -> None:
    assert canonical("") == ""
