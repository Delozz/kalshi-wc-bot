# Kalshi WC Bot

A quantitative trading bot for FIFA World Cup 2026 match outcome markets on [Kalshi](https://kalshi.com). It ingests multi-source football data, engineers time-safe features, runs a calibrated ML probability model, detects edge vs Kalshi implied probabilities, sizes bets via fractional Kelly, and executes live limit orders through the Kalshi REST API.

---

## How It Works

```
Ingest → Features → Model → Edge Detection → Kelly Sizing → Order → Settle → Log
```

1. **Ingest** — Pull live fixtures (API-Football), historical results (martj42 CSV), announced lineups, and live Kalshi market prices
2. **Features** — ELO ratings, rolling form, head-to-head stats, StatsBomb xG averages, Pinnacle no-vig implied probabilities, lineup strength delta
3. **Model** — Dixon-Coles goals model with ELO strength prior (default); XGBoost multi-class classifier (Home / Draw / Away) with Platt-scaling calibration as fallback. Switch with `MODEL_ENGINE=classifier`. Baseline logistic regression must be beaten before production use
4. **Edge Detection** — `edge = model_prob − kalshi_implied_prob`; threshold ≥ 4% required to generate a signal
5. **Kelly Sizing** — Half-Kelly with a hard 5%-of-bankroll cap per bet; calibration must precede sizing
6. **Execution** — Kalshi V2 limit orders (`POST /portfolio/events/orders`); demo paper-run required before prod
7. **Settlement** — Matches resolved via API-Football result; P&L posted to SQLite ledger
8. **Dashboard** — Rich CLI live view of bankroll, open positions, and current-cycle signals

---

## Data Sources

| Source | What It Provides | Auth |
|---|---|---|
| [football-data.co.uk](https://football-data.co.uk) | Historical results + closing odds CSVs (Bet365, Pinnacle) | None |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | 2022 WC event data + xG | None |
| [martj42/international-football-results](https://github.com/martj42/international-football-results) | Full international results history | None |
| [API-Football](https://www.api-sports.io) | Live fixtures, injuries, lineups, H2H | `API_FOOTBALL_KEY` |
| [The Odds API](https://the-odds-api.com) | Live sportsbook implied probabilities, no-vig fair odds | `ODDS_API_KEY` |
| [Kalshi REST API](https://kalshi.com) | Live market prices, order book, order execution, portfolio | RSA key pair |

All API responses are cached to `data/raw/` on first fetch (rate limits are tight on free tiers).

---

## Project Structure

```
kalshi-wc-bot/
├── ingestion/          One module per data source; all wrapped in try/except
│   ├── api_football.py     Live WC fixtures, lineups, H2H
│   ├── kalshi.py           Kalshi REST client (markets, orders, portfolio)
│   ├── odds_api.py         The Odds API implied probabilities
│   ├── statsbomb.py        StatsBomb xG open data
│   ├── international_results.py  martj42 full results history
│   └── football_data_co.py Historical odds CSVs
├── features/           Time-aware feature builders (always gated by cutoff_time)
│   ├── pipeline.py         Orchestrates all builders
│   ├── elo.py              ELO ratings + match delta
│   ├── form.py             Rolling form windows
│   ├── lineup.py           Lineup strength delta (home − away mean rating)
│   ├── odds_features.py    Pinnacle no-vig probabilities + line movement
│   ├── xg_features.py      StatsBomb xG rolling averages
│   └── h2h.py              Head-to-head win rate and goals average
├── model/              Training, calibration, evaluation
│   ├── train.py            Training entrypoint — saves versioned .pkl artifact
│   ├── baseline.py         Logistic regression baseline (must be beaten)
│   ├── xgboost_model.py    XGBoost multi-class classifier
│   ├── calibration.py      Platt scaling + reliability diagram
│   └── evaluate.py         Brier score, log loss, calibration curves
├── strategy/           Signal generation and risk management
│   ├── edge.py             Edge computation + lineup-adjusted probability
│   ├── kelly.py            Half-Kelly sizing with hard bet cap
│   ├── signal_gen.py       Full pipeline: fixtures → signals → orders
│   └── risk.py             Stop-loss, exposure caps, liquidity filters
├── execution/          Order lifecycle and portfolio state
│   ├── order_manager.py    Place, poll, cancel Kalshi limit orders (V2 API)
│   └── portfolio.py        Open positions, bankroll, exposure state
├── backtest/           Time-ordered backtesting engine
│   ├── engine.py           2018 dev / 2022 holdout event loop
│   ├── lookahead_guard.py  filter_data(df, cutoff) — raises on violations
│   ├── simulator.py        Simulated fills at historical closing prices
│   └── metrics.py          P&L, ROI, Sharpe, hit rate, max drawdown
├── dashboard/
│   └── app.py              Rich CLI dashboard — bankroll, signals, positions
├── scheduler/
│   └── jobs.py             APScheduler — odds refresh, signal gen, order placement
├── data/
│   ├── raw/                Cached API responses and CSVs (gitignored)
│   ├── processed/          Cleaned feature matrices in parquet (gitignored)
│   └── db.sqlite           Matches, features, signals, orders, bankroll log
└── tests/              Pytest suite — 91 tests, no network calls
```

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Fill in API keys in .env
```

### `.env` keys required

```
KALSHI_ENV=demo                    # "demo" for paper trading, "prod" for live

# Separate key pairs — switching KALSHI_ENV is all that's needed
KALSHI_DEMO_API_KEY=               # Demo account key ID
KALSHI_DEMO_API_SECRET=            # Demo RSA private key (PEM text or path)
KALSHI_PROD_API_KEY=               # Prod account key ID
KALSHI_PROD_API_SECRET=            # Prod RSA private key (PEM text or path)

API_FOOTBALL_KEY=                  # API-Football key
THE_ODDS_API_KEY=                  # The Odds API key

MODEL_ENGINE=dc                    # "dc" (Dixon-Coles, default) or "classifier"
INITIAL_BANKROLL_CENTS=5000        # Starting bankroll in cents (e.g. 5000 = $50)
LINEUP_WEIGHT=0.10                 # Lineup strength adjustment weight
DC_SQUAD_PRIOR_WEIGHT=0.5         # Squad-strength blend into DC ELO prior (0 = off)
```

> **Never commit `.env`** — it is gitignored. See `.env.example` for the template.

---

## Common Commands

```powershell
# Train model (dev set = 2018 WC)
python -m model.train

# Evaluate calibration
python -m model.evaluate --holdout dev

# Dry-run signal generation (no orders sent)
python -m scheduler.jobs --once

# Paper-trade on Kalshi demo environment (set KALSHI_ENV=demo in .env)
python -m scheduler.jobs --once --live-orders

# Go live on prod (requires KALSHI_ALLOW_PROD_ORDERS=1 in .env after clean demo run)
python -m scheduler.jobs --once --live-orders

# Start the continuous scheduler
python -m scheduler.jobs

# Launch live dashboard
python -m dashboard.app

# Run tests
pytest tests/ -v

# Lint + format check
ruff check .
black --check .
```

---

## Running 24/7 (Windows Task Scheduler)

The bot can run unattended via a Windows Scheduled Task that restarts automatically on crash and survives reboots.

```powershell
# Register the task (run once in elevated PowerShell)
# See scripts/run_scheduler.ps1 for the launcher used by the task

# Check task health
& ".\scripts\status.ps1"

# Stop the loop (keeps task registered for next time)
Stop-ScheduledTask -TaskName "KalshiWCBotLoop"

# Disable until manually re-enabled
Disable-ScheduledTask -TaskName "KalshiWCBotLoop"

# Re-enable
Enable-ScheduledTask -TaskName "KalshiWCBotLoop"
```

Logs are written to `data/logs/scheduler.log`. Sleep/hibernate must be disabled (`powercfg /change standby-timeout-ac 0`) — a sleeping PC freezes the loop. The PC must be powered on at tournament time; the task recovers automatically at next boot via the `AtStartup` trigger.

---

## Key Safety Rules

| Rule | What It Prevents |
|---|---|
| **No look-ahead** | Features must pass `lookahead_guard.filter_data(df, cutoff=kickoff_utc)` — violations are silent and lethal to backtest validity |
| **Holdout is sacred** | 2022 WC data is touched exactly once: final model evaluation only; never used for tuning |
| **Calibrate before Kelly** | Kelly sizing never runs on uncalibrated model probabilities — overconfidence causes catastrophic overbetting |
| **Half-Kelly + 5% cap** | `bet = min(kelly * 0.5, 0.05 * bankroll)` — hard limit, not a suggestion |
| **Demo before prod** | Full paper-trade run on Kalshi demo required before any prod orders |
| **Wrap every ingestion call** | One source failure must never crash the scheduler — all ingestion is `try/except` |

---

## Strategy Parameters

| Parameter | Value | Description |
|---|---|---|
| Minimum edge | 4% | `model_prob − kalshi_implied_prob` required to generate a signal |
| Kelly fraction | 0.5 | Half-Kelly applied to all sizing |
| Max bet size | 5% of bankroll | Hard cap per position |
| Max portfolio exposure | 20% of bankroll | Sum of all open position sizes |
| Stop-loss | 25% drawdown from peak | Halts all betting if breached |
| Max open positions | 5 | Concurrent open position limit |

---

## Backtesting

The backtest engine uses a strict time-ordered event loop with no look-ahead:

- **Development set:** 2018 FIFA World Cup
- **Holdout set:** 2022 FIFA World Cup (never touched during development or tuning)
- Features are always computed with `cutoff_time = kickoff_utc`
- Simulated fills use historical closing prices from football-data.co.uk

```powershell
python -m backtest.engine --tournament 2018
```

---

## Tech Stack

- **Python 3.11+** — type hints on every function
- **pandas, scikit-learn, XGBoost, optuna** — feature engineering and modeling
- **httpx** — async HTTP for all ingestion modules
- **APScheduler** — cron-style job scheduling during the live tournament
- **SQLite** — local persistence for signals, orders, and bankroll log
- **rich** — CLI dashboard
- **pytest** — 91 tests, all network-free
- **ruff + black** — linting and formatting
