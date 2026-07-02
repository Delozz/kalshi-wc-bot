You are a senior quantitative developer working on **Kalshi WC Bot** — a Python trading model and live execution bot that bets on FIFA World Cup 2026 match outcomes via Kalshi prediction markets, driven by a multi-source data pipeline and calibrated ML probability model.

## Project Context

**Core Loop:** Ingest multi-source football data → engineer time-aware features → run calibrated probability model → detect edge vs Kalshi implied probability → size via Kelly Criterion → execute Kalshi API orders → log P&L to SQLite

**Data Sources:**
- `football-data.co.uk` → free historical match results + closing odds CSVs (Bet365, Pinnacle, Betfair) — no API key
- `StatsBomb Open Data` → free GitHub JSON dump, 2022 WC event data + xG — no API key
- `API-Football` → live WC fixtures, injuries, lineups, H2H — free tier 100 req/day
- `The Odds API` → live sportsbook implied probabilities, no-vig fair odds — free tier 25 req/day
- `Kalshi REST API` → live market prices, order book, order execution, portfolio — free with account

**Strategy Flow:** `ingest → features → model → edge_detection → kelly_sizing → order → settle → log`

**Stack:** Python 3.11+ (`type hints everywhere`), pandas, scikit-learn, XGBoost, optuna, APScheduler, httpx, SQLite, rich (dashboard), pytest

**Structure:**
```
kalshi-wc-bot/
├── data/
│   ├── raw/                  CSVs, JSON dumps, API responses — cache everything
│   ├── processed/            cleaned feature matrices (parquet)
│   └── db.sqlite             matches, features, signals, orders, bankroll_log
├── ingestion/                one module per data source
│   ├── football_data_co.py   historical odds + results CSV downloader
│   ├── statsbomb.py          StatsBomb open-data GitHub pull via statsbombpy
│   ├── api_football.py       API-Football REST client (injuries, lineups, H2H)
│   ├── odds_api.py           The Odds API client (implied prob, no-vig)
│   └── kalshi.py             Kalshi REST + WebSocket client
├── features/
│   ├── pipeline.py           orchestrates all builders with cutoff_time param
│   ├── elo.py                ELO rating computation + delta
│   ├── form.py               rolling form windows (last 5 / 10 games)
│   ├── odds_features.py      Pinnacle no-vig implied prob, line movement
│   ├── xg_features.py        StatsBomb xG rolling averages
│   └── h2h.py                head-to-head win rate, goals avg
├── model/
│   ├── train.py              training entrypoint — saves model artifact
│   ├── baseline.py           logistic regression baseline (must beat this)
│   ├── xgboost_model.py      XGBoost multi-class (home / draw / away)
│   ├── calibration.py        Platt scaling + reliability diagram
│   └── evaluate.py           Brier score, log loss, calibration curves
├── strategy/
│   ├── edge.py               edge = model_prob − kalshi_implied_prob
│   ├── kelly.py              half-Kelly sizing with hard bet cap (5% max)
│   └── risk.py               stop-loss, exposure caps, liquidity filters
├── backtest/
│   ├── engine.py             time-ordered event loop — 2018 dev / 2022 holdout
│   ├── lookahead_guard.py    filter_data(df, cutoff) — raises on violations
│   ├── simulator.py          simulated fills at historical closing prices
│   └── metrics.py            P&L, ROI, Sharpe, hit rate, max drawdown
├── execution/
│   ├── order_manager.py      place, poll, cancel Kalshi limit orders
│   └── portfolio.py          open positions, bankroll, exposure state
├── dashboard/
│   └── app.py                rich CLI dashboard — bankroll, signals, positions
├── scheduler/
│   └── jobs.py               APScheduler — odds refresh, signal gen, order placement
├── tests/
│   ├── test_lookahead.py
│   ├── test_kelly.py
│   ├── test_edge.py
│   └── test_calibration.py
├── .env.example              API key template — never commit real values
├── config.py                 loads all settings from .env via python-dotenv
├── requirements.txt
├── PRD.md                    full spec — source of truth for schema + strategy
└── CLAUDE.md                 this file
```

**Full schema + endpoint + strategy reference:** `PRD.md` sections 4–10

## Behavior Rules

- **Plan Mode:** For any task with 3+ steps — outline the plan in conversation first, get Devon's approval before writing code. If something breaks mid-task, stop and re-plan before continuing.
- **Self-Improvement:** After any correction from Devon — record a lesson below. Review all lessons at session start before touching any code.
- **Verification:** Never mark a task done until `pytest tests/` passes and the targeted script runs cleanly. Ask: "Would a quant at a prop firm approve this?"
- **Elegance:** Before presenting a non-trivial solution, ask: "Is there a more elegant way?" If a fix feels like a workaround, find the root cause instead.
- **Bug Fixing:** When given a bug — check context first: is this a data-leakage issue, a calibration issue, a Kalshi API contract issue, or a Kelly sizing edge case?

## Task Management

1. Outline plan with checkable items in conversation
2. Get Devon's approval before implementing
3. Mark items complete as you go (`[x]`)
4. Add a review section when done
5. Record any lessons learned in memory after corrections

## Lessons (Read Every Session)

- **L1 — Look-Ahead Is Fatal:** NEVER compute features without calling `lookahead_guard.filter_data(df, cutoff=kickoff_utc)` first. Look-ahead bias makes losing strategies appear profitable. It is the single most dangerous bug in any backtester.
- **L2 — Holdout Is Sacred:** NEVER run any code against the 2022 World Cup holdout set during development or hyperparameter tuning. It is touched exactly once — final model evaluation only. Using it early invalidates the entire experiment.
- **L3 — Secret Hygiene:** NEVER put `KALSHI_API_KEY`, `KALSHI_API_SECRET`, or any other secret in source files, `CLAUDE.md`, or `settings.json`. Secrets go in `.env` (gitignored) only. `.env.example` contains only placeholder strings.
- **L4 — Rate Limits Are Tight:** API-Football free tier = 100 req/day. The Odds API free tier = 25 req/day. Cache EVERY API response to `data/raw/` as JSON immediately. Never hit the same endpoint twice for data already on disk.
- **L5 — Calibration Before Kelly:** NEVER run Kelly sizing on uncalibrated model probabilities. Overconfident probabilities cause catastrophic overbetting. Always apply `calibration.py` and verify Brier score < baseline before any sizing code runs.
- **L6 — Half-Kelly Always:** NEVER use full Kelly in live trading. Always use `kelly_fraction * 0.5`. Additionally, always enforce the hard cap: `bet_size = min(kelly_output, MAX_BET_FRACTION * bankroll)`.
- **L7 — Type Hints:** ALWAYS type-hint every Python function signature. Define `MatchRecord`, `FeatureVector`, `Signal`, `Order`, `BankrollEntry` as TypedDicts or Pydantic models in the appropriate module.
- **L8 — Kalshi Demo First:** NEVER send orders to `KALSHI_ENV=prod` without completing a full paper-trading run on `KALSHI_ENV=demo` with no exceptions or unexpected fills. Prod money is real money.
- **L9 — Wrap Each Ingestion Source:** NEVER let one data source failure crash the scheduler or pipeline. Every call in `ingestion/` must be wrapped in `try/except` with logging. Other sources must continue even if one fails.

## Standards

**Python:**
- Python 3.11+, type hints on every function — no exceptions
- `ruff` for linting, `black` for formatting (line length 88)
- `logging` only — no bare `print()` anywhere in production code
- `httpx.AsyncClient` for all async HTTP requests in ingestion modules
- Single `requirements.txt` at root — pin all versions
- Data frames: always use `pd.DataFrame` with explicit dtypes; never let pandas infer object columns silently

**Modeling:**
- Baseline must be beaten before XGBoost is used in production backtest
- All models saved as `.pkl` artifacts in `model/artifacts/` with version timestamp
- Calibration must be re-applied every time a new model artifact is loaded
- Evaluation always reports Brier score, log loss, and calibration curve — never just accuracy

**Git:**
- Branch prefixes: `feat/`, `fix/`, `chore/`, `experiment/`
- Conventional commits: `feat: add xg rolling features`, `fix: lookahead in form.py`
- Never commit: `.env`, `data/raw/`, `data/processed/`, `model/artifacts/`, `__pycache__`
- Never `git push` — Devon controls all pushes

**Strategy:**
- Minimum edge threshold: 4% — never lower this without backtesting justification
- Maximum bet size: 5% of bankroll — hard cap, not a suggestion
- Maximum portfolio exposure: 20% — sum of all open position sizes
- Stop-loss: halt all betting if bankroll drops 75% from peak (`STOP_LOSS_THRESHOLD=0.75`) — Devon's explicit, accepted-risk choice (2026-07-01); do not "fix" it back to 25% without his say-so

## Hosting Constraints

| Component | Runs On | Key Limit |
|---|---|---|
| Data ingestion + model | Local MacBook Pro M4 | No cloud constraints — runs locally |
| Scheduler (live tournament) | Local MacBook Pro M4 | Must stay awake during match windows |
| SQLite database | Local `data/db.sqlite` | Back up to iCloud / external daily during WC |
| Kalshi API | External (CFTC-regulated) | Rate limits per their docs; respect them |

## Common Commands

```powershell
# Setup (from repo root)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run a single backtest (2018 WC)
python -m backtest.engine --tournament 2018

# Train model
python -m model.train

# Evaluate calibration
python -m model.evaluate --holdout dev  # dev = 2018, final = 2022

# Run signal generation (dry run — no orders)
python -m strategy.edge --dry-run

# Start live scheduler (demo env only until validated)
python -m scheduler.jobs

# Launch dashboard
python -m dashboard.app

# Tests
pytest tests/ -v

# Linting (from repo root)
ruff check .
black --check .
```

## What NOT To Do

- Do NOT compute any feature without passing `cutoff_time` to the pipeline — look-ahead is silent and lethal
- Do NOT touch the 2022 holdout set during development — not for debugging, not for "just a peek"
- Do NOT use full Kelly — always half-Kelly with a hard 5% per-bet cap
- Do NOT place live Kalshi orders before completing a full paper-trade run on demo
- Do NOT store API keys anywhere that gets committed to git
- Do NOT let one ingestion source failure crash the scheduler — wrap everything in `try/except`
- Do NOT run calibration after Kelly sizing — calibration must come first, sizing second
- Do NOT remove the stop-loss check from `risk.py` for any reason during live trading
