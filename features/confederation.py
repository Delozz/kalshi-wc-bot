"""Confederation strength correction (features/confederation.py).

ELO (and the Dixon-Coles ratings the ELO prior seeds) is computed over a single global pool,
but national teams play overwhelmingly *within* their own confederation — qualifiers, regional
cups — so each confederation's ratings are only loosely anchored to the others by the handful
of cross-confederation games. The result is **inter-confederation drift**: an AFC side that
farms goals on weak regional opponents accrues an ELO that doesn't transfer when it faces
UEFA/CONMEBOL. This is exactly what made the model overprice Japan-over-Brazil, Iran, etc.

Measured on neutral-venue cross-confederation matches (2010+, both sides pooled), the mean
ELO "surprise" (actual − expected result) per confederation maps to these additive ELO
offsets — strong confederations are *under*-rated by raw ELO, weak ones *over*-rated:

    CONMEBOL +64   UEFA +62   CAF −10   CONCACAF −70   AFC −79

Applied as an *additive per-team ELO offset*, the correction is automatically venue-correct
for the use case: two same-confederation teams shift equally so it cancels (no effect on an
all-AFC group game), while a cross-confederation matchup gets the full differential. The
offset feeds two places in ``strategy/signal_gen``: a post-model probability tilt
(``strategy/edge.apply_confederation_prior``) and the favorite/ELO-gap used by the risk
filters, so both the sizing and the guardrails see true cross-confederation strength.

OFC has too few teams/games to measure directly; it is set to the CONCACAF level by analogy
(clearly among the weakest) so an Oceania underdog isn't left uncorrected. Any team absent
from the map gets a 0.0 offset — a zero-impact fallback, never a fabricated adjustment.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Additive ELO offsets by confederation (see module docstring for derivation).
CONFEDERATION_ELO_OFFSET: dict[str, float] = {
    "CONMEBOL": 64.0,
    "UEFA": 62.0,
    "CAF": -10.0,
    "CONCACAF": -70.0,
    "AFC": -79.0,
    "OFC": -70.0,  # un-measured; set to the CONCACAF level by analogy (weakest pool)
}

# Team -> confederation. Keys are the canonical martj42 names the pipeline uses. Covers the
# 2026 WC field plus frequent cross-confederation opponents; unlisted teams fall back to 0.0.
CONFEDERATION: dict[str, str] = {
    # --- UEFA -------------------------------------------------------------------------
    "England": "UEFA",
    "France": "UEFA",
    "Spain": "UEFA",
    "Germany": "UEFA",
    "Italy": "UEFA",
    "Netherlands": "UEFA",
    "Portugal": "UEFA",
    "Belgium": "UEFA",
    "Croatia": "UEFA",
    "Denmark": "UEFA",
    "Switzerland": "UEFA",
    "Sweden": "UEFA",
    "Poland": "UEFA",
    "Serbia": "UEFA",
    "Austria": "UEFA",
    "Wales": "UEFA",
    "Ukraine": "UEFA",
    "Czech Republic": "UEFA",
    "Norway": "UEFA",
    "Scotland": "UEFA",
    "Russia": "UEFA",
    "Turkey": "UEFA",
    "Hungary": "UEFA",
    "Republic of Ireland": "UEFA",
    "Romania": "UEFA",
    "Greece": "UEFA",
    "Slovakia": "UEFA",
    "Slovenia": "UEFA",
    "Iceland": "UEFA",
    "Finland": "UEFA",
    "Albania": "UEFA",
    "Bulgaria": "UEFA",
    "North Macedonia": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Montenegro": "UEFA",
    "Northern Ireland": "UEFA",
    "Georgia": "UEFA",
    "Israel": "UEFA",
    "Kosovo": "UEFA",
    "Luxembourg": "UEFA",
    "Cyprus": "UEFA",
    "Armenia": "UEFA",
    "Azerbaijan": "UEFA",
    "Kazakhstan": "UEFA",
    "Belarus": "UEFA",
    "Moldova": "UEFA",
    "Estonia": "UEFA",
    "Latvia": "UEFA",
    "Lithuania": "UEFA",
    "Malta": "UEFA",
    "Andorra": "UEFA",
    "San Marino": "UEFA",
    "Liechtenstein": "UEFA",
    "Gibraltar": "UEFA",
    "Faroe Islands": "UEFA",
    # --- CONMEBOL ---------------------------------------------------------------------
    "Brazil": "CONMEBOL",
    "Argentina": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Chile": "CONMEBOL",
    "Peru": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    # --- CONCACAF ---------------------------------------------------------------------
    "United States": "CONCACAF",
    "Mexico": "CONCACAF",
    "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF",
    "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF",
    "Panama": "CONCACAF",
    "El Salvador": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF",
    "Guatemala": "CONCACAF",
    "Haiti": "CONCACAF",
    "Curacao": "CONCACAF",
    "Suriname": "CONCACAF",
    "Nicaragua": "CONCACAF",
    "Cuba": "CONCACAF",
    # --- AFC --------------------------------------------------------------------------
    "Japan": "AFC",
    "South Korea": "AFC",
    "Iran": "AFC",
    "Australia": "AFC",
    "Saudi Arabia": "AFC",
    "Qatar": "AFC",
    "Iraq": "AFC",
    "Uzbekistan": "AFC",
    "United Arab Emirates": "AFC",
    "China PR": "AFC",
    "Jordan": "AFC",
    "Oman": "AFC",
    "Bahrain": "AFC",
    "Syria": "AFC",
    "Vietnam": "AFC",
    "Thailand": "AFC",
    "Palestine": "AFC",
    "Lebanon": "AFC",
    "India": "AFC",
    "Kuwait": "AFC",
    "Kyrgyzstan": "AFC",
    "Tajikistan": "AFC",
    "Turkmenistan": "AFC",
    "North Korea": "AFC",
    "Indonesia": "AFC",
    "Malaysia": "AFC",
    # --- CAF --------------------------------------------------------------------------
    "Egypt": "CAF",
    "Nigeria": "CAF",
    "Senegal": "CAF",
    "Morocco": "CAF",
    "Ghana": "CAF",
    "Cameroon": "CAF",
    "Algeria": "CAF",
    "Tunisia": "CAF",
    "Ivory Coast": "CAF",
    "Mali": "CAF",
    "DR Congo": "CAF",
    "South Africa": "CAF",
    "Burkina Faso": "CAF",
    "Cape Verde": "CAF",
    "Guinea": "CAF",
    "Zambia": "CAF",
    "Angola": "CAF",
    "Gabon": "CAF",
    "Benin": "CAF",
    "Uganda": "CAF",
    "Mauritania": "CAF",
    "Equatorial Guinea": "CAF",
    "Madagascar": "CAF",
    "Namibia": "CAF",
    "Mozambique": "CAF",
    "Sudan": "CAF",
    "Tanzania": "CAF",
    "Kenya": "CAF",
    "Zimbabwe": "CAF",
    "Ethiopia": "CAF",
    "Congo": "CAF",
    "Togo": "CAF",
    "Libya": "CAF",
    "Comoros": "CAF",
    "Gambia": "CAF",
    "Guinea-Bissau": "CAF",
    # --- OFC --------------------------------------------------------------------------
    "New Zealand": "OFC",
    "New Caledonia": "OFC",
    "Tahiti": "OFC",
    "Fiji": "OFC",
    "Solomon Islands": "OFC",
    "Vanuatu": "OFC",
    "Papua New Guinea": "OFC",
}


def confederation_of(team: str) -> str | None:
    """The team's confederation code, or ``None`` if it isn't in the map."""
    return CONFEDERATION.get(team)


def offset_for(team: str) -> float:
    """Additive ELO offset for a team's confederation; 0.0 for unmapped teams.

    The 0.0 fallback is intentional: an unknown team neither lifts nor drags the matchup,
    so the correction degrades to a no-op rather than fabricating a strength adjustment.
    """
    conf = CONFEDERATION.get(team)
    if conf is None:
        return 0.0
    return CONFEDERATION_ELO_OFFSET.get(conf, 0.0)


def elo_delta(home: str, away: str) -> float:
    """Home-minus-away confederation ELO offset (signed ELO points).

    Positive favours the home side. For two same-confederation teams this is exactly 0.0
    (the offsets cancel), so an intra-confederation matchup is never adjusted; a cross-
    confederation matchup gets the full strength differential.
    """
    return offset_for(home) - offset_for(away)
