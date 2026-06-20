"""Team-name normalization (features/teams.py).

Different sources spell national teams differently (API-Football "USA" vs martj42
"United States"). Our ELO ratings are keyed by martj42 names, so that is the canonical
form: every team name is normalized to it before a rating lookup or market match. The
alias map is deliberately small and not exhaustive — extend it as real API responses
surface mismatches.
"""

from __future__ import annotations

# Lowercased source spelling -> canonical (martj42) name.
_ALIASES: dict[str, str] = {
    "usa": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "korea dpr": "North Korea",
    "ir iran": "Iran",
    "china pr": "China",
    "czechia": "Czech Republic",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "cabo verde": "Cape Verde",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "congo dr": "DR Congo",
    "republic of ireland": "Ireland",
}


def canonical(name: str) -> str:
    """Map a team name to its canonical (martj42) spelling.

    Unknown names pass through trimmed but otherwise unchanged (most sources already
    agree with martj42), so this is safe to call on any input.
    """
    if not name:
        return name
    trimmed = name.strip()
    return _ALIASES.get(trimmed.lower(), trimmed)
