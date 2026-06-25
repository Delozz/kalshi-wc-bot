"""Central configuration — loads all settings from .env (never hardcode secrets, L3)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "model" / "artifacts"

KalshiEnv = Literal["demo", "prod"]


def _get_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    return raw if raw not in (None, "") else default


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw not in (None, "") else default


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot. Secrets may be empty during Phase 1 dev."""

    kalshi_api_key: str
    kalshi_api_secret: str
    api_football_key: str
    the_odds_api_key: str
    kalshi_env: KalshiEnv
    min_edge_threshold: float
    max_bet_fraction: float
    max_portfolio_exposure: float
    kelly_fraction: float
    stop_loss_threshold: float
    initial_bankroll_cents: int
    lineup_weight: float
    squad_weight: float
    db_path: Path
    log_level: str


def load_settings() -> Settings:
    """Build a Settings snapshot from the current environment."""
    env_raw = _get_str("KALSHI_ENV", "demo")
    kalshi_env: KalshiEnv = "prod" if env_raw == "prod" else "demo"
    return Settings(
        kalshi_api_key=_get_str("KALSHI_API_KEY", ""),
        kalshi_api_secret=_get_str("KALSHI_API_SECRET", ""),
        api_football_key=_get_str("API_FOOTBALL_KEY", ""),
        the_odds_api_key=_get_str("THE_ODDS_API_KEY", ""),
        kalshi_env=kalshi_env,
        min_edge_threshold=_get_float("MIN_EDGE_THRESHOLD", 0.04),
        max_bet_fraction=_get_float("MAX_BET_FRACTION", 0.05),
        max_portfolio_exposure=_get_float("MAX_PORTFOLIO_EXPOSURE", 0.20),
        kelly_fraction=_get_float("KELLY_FRACTION", 0.5),
        stop_loss_threshold=_get_float("STOP_LOSS_THRESHOLD", 0.25),
        initial_bankroll_cents=_get_int("INITIAL_BANKROLL_CENTS", 20000),
        # Max fractional swing applied to a model probability when a fully-rated lineup
        # is available (lineup_delta in [-1, 1]). 0 disables lineup adjustment entirely.
        lineup_weight=_get_float("LINEUP_WEIGHT", 0.10),
        # Strength of the always-on squad-strength prior (squad_delta in [-1, 1] tilts the
        # full H/D/A vector toward the stronger squad). 4.0 chosen from a weight-sweep
        # replay against real bets — a meaningful but calibration-safe tilt given how
        # compressed national-team ratings are. 0 disables the squad prior entirely.
        squad_weight=_get_float("SQUAD_WEIGHT", 4.0),
        db_path=Path(_get_str("DB_PATH", "data/db.sqlite")),
        log_level=_get_str("LOG_LEVEL", "INFO"),
    )


settings: Settings = load_settings()


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once. Use logging, never print() in production code."""
    logging.basicConfig(
        level=level or settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dirs() -> None:
    """Create the data/artifact directories if they do not yet exist."""
    for directory in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
