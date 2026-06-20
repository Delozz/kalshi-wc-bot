# PRD — Kalshi World Cup 2026 Trading Model / Bot

**Version:** 1.0  
**Author:** Devon Lopez  
**Status:** Planning  
**Last Updated:** June 2026

---

## 1. Overview

A quantitative trading bot that places profitable bets on Kalshi prediction market contracts for the 2026 FIFA World Cup (June–July 2026, hosted in USA/Canada/Mexico). The system ingests multi-source football data, builds calibrated match-outcome probability models, detects pricing edges against Kalshi's implied probabilities, sizes positions via Kelly Criterion, and executes orders through the Kalshi REST API — all with a full backtesting framework validated on 2018 and 2022 World Cup data before any live capital is deployed.

**Primary goal:** Achieve positive expected value (EV) over the tournament with controlled drawdown.  
**Secondary goal:** Build a reusable, modular quant pipeline Devon can extend to future prediction markets.

---

## 2. Goals & Non-Goals

### Goals
- Ingest and normalize data from all five source categories (odds, team stats, Kalshi markets, news/injuries, FIFA rankings)
- Train and calibrate a match-outcome probability model on historical World Cup data
- Backtest the full strategy on 2018 and 2022 data with no look-ahead bias
- Detect positive-edge opportunities: `model_prob − kalshi_implied_prob > threshold`
- Size bets using fractional Kelly with hard per-bet and total-exposure caps
- Execute live orders via Kalshi REST API during the 2026 tournament
- Log all decisions, P&L, and model performance to a local SQLite database
- Display a real-time dashboard (CLI or simple web UI) showing bankroll, open positions, and P&L

### Non-Goals
- Trading markets outside FIFA World Cup 2026 (v1 scope only)
- High-frequency or in-play trading (pre-match only for v1)
- Multi-user support or cloud deployment (runs locally on MacBook Pro M4)
- Mobile app or public-facing UI
- Social/copy trading features

---

## 3. System Architecture

```
kalshi-wc-bot/
├── data/
│   ├── raw/                  downloaded CSVs, JSON dumps, API responses
│   ├── processed/            cleaned, normalized feature matrices
│   └── db.sqlite             all logs, bets, model outputs
├── ingestion/                one module per data source
│   ├── football_data_co.py   football-data.co.uk CSV downloader
│   ├── statsbomb.py          StatsBomb open-data GitHub pull
│   ├── api_football.py       API-Football REST client
│   ├── odds_api.py           The Odds API client
│   └── kalshi.py             Kalshi REST + WebSocket client
├── features/
│   ├── pipeline.py           orchestrates all feature builders
│   ├── elo.py                ELO rating computation + delta
│   ├── form.py               rolling form windows (last N games)
│   ├── odds_features.py      implied prob, no-vig fair odds, line movement
│   ├── xg_features.py        expected goals rolling averages
│   └── h2h.py                head-to-head historical record
├── model/
│   ├── train.py              training entrypoint — outputs model artifact
│   ├── baseline.py           logistic regression baseline
│   ├── xgboost_model.py      XGBoost match-outcome classifier
│   ├── calibration.py        Platt scaling, isotonic regression
│   └── evaluate.py           Brier score, log loss, calibration curve
├── strategy/
│   ├── edge.py               edge = model_prob − market_implied_prob
│   ├── kelly.py              fractional Kelly + exposure caps
│   └── risk.py               max-per-bet, max-total-exposure, stop-loss
├── backtest/
│   ├── engine.py             time-ordered event loop
│   ├── lookahead_guard.py    enforces data[timestamp < match_time]
│   ├── simulator.py          simulated order fills at historical prices
│   └── metrics.py            P&L, ROI, Sharpe, hit rate, max drawdown
├── execution/
│   ├── order_manager.py      place, cancel, track Kalshi orders
│   └── portfolio.py          open positions, bankroll, exposure tracking
├── dashboard/
│   └── app.py                Rich CLI dashboard (or optional Flask UI)
├── scheduler/
│   └── jobs.py               APScheduler — daily data refresh + signal scan
├── tests/
│   ├── test_lookahead.py
│   ├── test_kelly.py
│   ├── test_edge.py
│   └── test_calibration.py
├── .env.example              API key template — never commit real values
├── config.py                 central config (loaded from .env)
├── requirements.txt
├── PRD.md                    this file — source of truth
└── CLAUDE.md                 AI coding assistant instructions
```

---

## 4. Data Sources

### 4.1 Historical Match Odds — football-data.co.uk
- **URL:** `https://www.football-data.co.uk/`
- **Cost:** Free — direct CSV download, no API key
- **What we use:** Pre-match and closing odds from Bet365, Pinnacle, Betfair across 25+ leagues, seasons 2000/01–2025/26, plus international tournament data
- **Format:** CSV files per league per season
- **Use case:** Backtesting implied probabilities; training set for historical edge detection
- **Ingestion module:** `ingestion/football_data_co.py`
- **Key columns:** `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `B365H`, `B365D`, `B365A`, `PSH`, `PSD`, `PSA` (Pinnacle), `BFH`, `BFD`, `BFA` (Betfair)
- **Rate limit:** None — static file downloads

### 4.2 Deep Match Stats + xG — StatsBomb Open Data
- **URL:** `https://github.com/statsbomb/open-data`
- **Cost:** Free — no authentication required
- **What we use:** 2022 FIFA World Cup full event data (competition_id: 43, season_id: 106), 360-degree tracking data, every pass/shot/duel, xG per match
- **Format:** JSON files on GitHub; Python package `statsbombpy` for easy access
- **Use case:** Advanced features — xG rolling averages, shot quality, pressing intensity
- **Ingestion module:** `ingestion/statsbomb.py`
- **Install:** `pip install statsbombpy`
- **Key data:** `sb.matches(competition_id=43, season_id=106)`, `sb.events(match_id=...)`
- **Rate limit:** GitHub raw content — no strict limit; cache locally after first pull

### 4.3 Live Fixtures, Injuries, Lineups — API-Football
- **URL:** `https://www.api-football.com/`
- **Cost:** Free tier — 100 requests/day, no credit card required; all endpoints accessible
- **What we use:** 2026 World Cup fixtures (`league=1&season=2026`), injuries, player ratings, lineups, suspensions, head-to-head records
- **Format:** JSON REST API
- **Auth:** `x-apisports-key` header
- **Use case:** Real-time pre-match signals during the live tournament
- **Ingestion module:** `ingestion/api_football.py`
- **Key endpoints:**
  - `GET /fixtures?league=1&season=2026` — all 104 WC matches
  - `GET /injuries?league=1&season=2026` — injury list per match week
  - `GET /fixtures/headtohead?h2h=TEAM1-TEAM2` — H2H history
  - `GET /fixtures/players?fixture={id}` — player ratings per match
- **Rate limit:** 100 req/day — cache all responses to SQLite immediately

### 4.4 Live Odds + Implied Probability — The Odds API
- **URL:** `https://the-odds-api.com/`
- **Cost:** Free tier — 25 requests/day (no credit card); paid starts at ~$79/mo for 500 req
- **What we use:** Real-time pre-match odds from 50+ bookmakers; fair-odds endpoint strips vig for clean true probability
- **Format:** JSON REST API
- **Auth:** `apiKey` query param
- **Use case:** Computing market-implied probability to compare against model output
- **Ingestion module:** `ingestion/odds_api.py`
- **Key endpoints:**
  - `GET /v4/sports/soccer_fifa_world_cup/odds?regions=us&markets=h2h&oddsFormat=decimal`
  - Returns Pinnacle, Bet365, DraftKings, FanDuel lines per match
- **Rate limit:** 25 req/day free — each request returns all bookmakers for all live matches, so 1–2 calls per match day is sufficient during the WC
- **Notes:** The Odds API is also used by Polymarket traders to benchmark prediction market contracts vs sportsbook pricing — directly analogous to our Kalshi use case

### 4.5 Kalshi Market Data — Kalshi REST API
- **URL:** `https://external-api.kalshi.com/trade-api/v2`
- **Cost:** Free for all verified Kalshi users; 0% trading fees currently
- **What we use:** Active World Cup market listings, real-time prices, order book, historical settled markets
- **Format:** JSON REST API + WebSocket for live updates
- **Auth:** Public endpoints require no auth; trading endpoints require API key (`KALSHI_API_KEY`)
- **Use case:** Market implied probability (YES price = implied prob), order placement, portfolio tracking
- **Ingestion module:** `ingestion/kalshi.py`
- **Key endpoints:**
  - `GET /markets?status=open&series_ticker=KXWC26` — active WC markets
  - `GET /markets/{ticker}/orderbook` — live order book
  - `GET /markets/{ticker}/history` — price history for a market
  - `POST /orders` — place an order (authenticated)
  - `GET /portfolio/positions` — open positions
  - `GET /portfolio/balance` — current bankroll
- **WebSocket:** `wss://external-api.kalshi.com/trade-api/ws/v2` — subscribe to real-time price ticks
- **Historical data caveat:** History is fragmented across endpoints; write a polling script that runs throughout the tournament and caches everything to SQLite

### 4.6 News & Sentiment — API-Football Injuries + Web
- **Cost:** Covered by API-Football free tier (see 4.3)
- **What we use:** Injury and suspension updates from API-Football; headline scraping from Google News RSS for major injury reports during the tournament
- **Ingestion:** Injury data pulled via `ingestion/api_football.py`; optional RSS scraper in `ingestion/news_rss.py` using standard `feedparser` (no API key)
- **Google News RSS:** `https://news.google.com/rss/search?q=FIFA+World+Cup+injury&hl=en`

---

## 5. Feature Engineering

All features are built by `features/pipeline.py` with a mandatory `cutoff_time` parameter. Every feature function must filter source data to `timestamp < cutoff_time` — this is enforced by `backtest/lookahead_guard.py`.

### 5.1 ELO Ratings (`features/elo.py`)
- Compute running ELO for every national team from 2000–present using `football_data_co.py` results
- Feature: `elo_delta = home_elo − away_elo`
- Starting ELO: 1500 for all teams; K-factor: 32 for group stage, 40 for knockouts
- Update after each match result

### 5.2 Rolling Form (`features/form.py`)
- `form_5_home`: points per game in last 5 matches for home team
- `form_5_away`: same for away team
- `form_10_home`, `form_10_away`: same over last 10
- `goals_scored_5_home`, `goals_conceded_5_home`: attacking/defensive form
- All computed as of `cutoff_time`

### 5.3 Implied Probability from Odds (`features/odds_features.py`)
- Convert Pinnacle decimal odds to no-vig implied probability:
  - `raw_home = 1 / PSH`, `raw_draw = 1 / PSD`, `raw_away = 1 / PSA`
  - `overround = raw_home + raw_draw + raw_away`
  - `fair_home = raw_home / overround` (remove vig)
- Feature: `pinnacle_implied_home`, `pinnacle_implied_draw`, `pinnacle_implied_away`
- These are the strongest single features — Pinnacle is the sharpest book globally

### 5.4 Expected Goals (`features/xg_features.py`)
- Source: StatsBomb open data (2022 WC, and club football for teams with data)
- `xg_for_5`: rolling 5-match xG for (attack quality)
- `xg_against_5`: rolling 5-match xG against (defense quality)
- `xg_delta`: `xg_for − xg_against`

### 5.5 Head-to-Head (`features/h2h.py`)
- `h2h_home_win_rate`: home team's win rate in last 5 H2H meetings
- `h2h_goals_avg`: average total goals in recent H2H

### 5.6 FIFA Rankings
- `fifa_rank_home`, `fifa_rank_away`: official monthly FIFA ranking at match time
- `fifa_rank_delta`: home rank − away rank (lower = better)
- Source: Scraped from `https://www.fifa.com/fifa-world-ranking/` monthly snapshots

---

## 6. Model

### 6.1 Target Variable
- `outcome`: 0 = home win, 1 = draw, 2 = away win
- Model outputs `P(home_win)`, `P(draw)`, `P(away_win)` — must sum to 1.0

### 6.2 Training Data
- **Source:** football-data.co.uk international fixtures + past World Cups
- **Train set:** All international match data 2000–2017 (excludes 2018 and 2022)
- **Validation set:** 2018 World Cup (64 matches)
- **Holdout set:** 2022 World Cup (64 matches) — touch only once at final eval

### 6.3 Models (in development order)

**Baseline — Logistic Regression** (`model/baseline.py`)
- Features: `elo_delta`, `pinnacle_implied_home`, `pinnacle_implied_draw`, `fifa_rank_delta`
- Serves as the floor — any complex model must beat this on Brier score

**Primary — XGBoost Classifier** (`model/xgboost_model.py`)
- All features from Section 5
- Multi-class output: `[P_home, P_draw, P_away]`
- Hyperparameter tuning via `optuna`

### 6.4 Calibration (`model/calibration.py`)
- Apply Platt scaling (logistic regression on model outputs) to calibrate probabilities
- Measure calibration with reliability diagrams and Brier score
- **Critical:** A model that says 70% must be right ~70% of the time. Uncalibrated models destroy Kelly sizing.

### 6.5 Evaluation Metrics (`model/evaluate.py`)
- **Brier Score:** primary metric (lower = better; random = 0.333 for 3 outcomes)
- **Log Loss:** penalizes confident wrong predictions
- **Calibration curve (reliability diagram):** plot predicted vs actual frequency per decile
- **ROI on holdout:** simulate flat-bet on highest-confidence outcome, measure return vs closing odds

---

## 7. Backtesting Framework

### 7.1 Engine (`backtest/engine.py`)
- Time-ordered loop: iterate matches chronologically by `kickoff_datetime`
- For each match:
  1. Call `lookahead_guard.filter(data, cutoff=kickoff_datetime)`
  2. Build features via `features/pipeline.py`
  3. Get model probabilities
  4. Compare to Kalshi/historical market implied probability
  5. Compute edge; skip if below threshold
  6. Size bet via Kelly
  7. Log simulated order to `data/db.sqlite`
  8. After match result: settle bet, update bankroll

### 7.2 Look-Ahead Guard (`backtest/lookahead_guard.py`)
- Single function: `filter_data(df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame`
- Raises `LookAheadError` if any row has `timestamp >= cutoff`
- Called at the start of every feature computation — non-negotiable

### 7.3 Transaction Cost Model
- Kalshi spread: assume 1–2 cent spread (YES ask − YES bid) per contract
- Apply as a cost on entry: `effective_price = ask_price` (not mid)
- Currently 0% Kalshi fees — but model the spread conservatively

### 7.4 Performance Metrics (`backtest/metrics.py`)

| Metric | Formula | Target |
|---|---|---|
| Total ROI | `(final_bankroll − initial) / initial` | > 0% |
| Sharpe Ratio | `mean(daily_returns) / std(daily_returns) × √252` | > 0.5 |
| Hit Rate | `winning_bets / total_bets` | > 45% |
| Max Drawdown | `max(peak − trough) / peak` | < 30% |
| Avg Edge | Mean of `model_prob − market_prob` on placed bets | > 3% |
| Bets Placed | Count of bets above threshold | Depends on WC schedule |

### 7.5 Backtest Runs
1. **Development backtest:** 2018 World Cup (can iterate freely)
2. **Final holdout:** 2022 World Cup (run once; if it fails, model needs rework — not parameter tweaks)

---

## 8. Strategy

### 8.1 Edge Detection (`strategy/edge.py`)
```python
def compute_edge(model_prob: float, kalshi_yes_price: float) -> float:
    """
    kalshi_yes_price is in cents (e.g., 0.62 = 62% implied probability).
    Returns edge as a decimal (0.10 = 10% edge).
    """
    return model_prob - kalshi_yes_price
```
- **Minimum edge threshold:** 0.04 (4%) to place any bet — below this, noise dominates
- **Markets to trade:** Match result (home/draw/away), group advancement, tournament winner (lower priority)

### 8.2 Kelly Criterion (`strategy/kelly.py`)
```
f* = (edge × b) / b    [simplified for binary outcomes where b = (1/price) − 1]
f* = (p × b − q) / b   where p = model_prob, q = 1 − p, b = net odds
```
- Use **half-Kelly** (`f* / 2`) to account for model uncertainty
- Hard cap: **5% of bankroll per bet** regardless of Kelly output
- **Never exceed:** 20% total portfolio exposure at any time

### 8.3 Risk Controls (`strategy/risk.py`)
- **Stop-loss:** Pause all betting if bankroll drops 25% from peak
- **Market concentration:** No more than 3 open positions simultaneously
- **Min liquidity:** Only bet into markets with at least $5,000 open interest
- **Stale signal:** Cancel signal if market price moves > 3 cents before order fills

---

## 9. Execution

### 9.1 Order Manager (`execution/order_manager.py`)
- Place limit orders at the ask price (never market orders)
- Poll fill status every 30 seconds; cancel and re-evaluate after 5 minutes unfilled
- Log every order attempt, fill, and cancel to `data/db.sqlite`

### 9.2 Portfolio Tracker (`execution/portfolio.py`)
- Maintains in-memory + SQLite state: open positions, bankroll, daily P&L
- Syncs with Kalshi `/portfolio/positions` and `/portfolio/balance` at startup

### 9.3 Pre-Match Workflow
- T-24h: Pull API-Football for injuries/suspensions/lineups
- T-2h: Fetch final odds snapshot from The Odds API + Kalshi prices
- T-1h: Run model, compute edge, generate signals
- T-30min: Place orders if signals pass all risk checks
- After match: Settle positions, update bankroll, log outcome

---

## 10. Database Schema (SQLite)

### `matches`
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | `{home}_{away}_{date}` |
| `home_team` | TEXT | |
| `away_team` | TEXT | |
| `kickoff_utc` | DATETIME | |
| `stage` | TEXT | group / round_of_16 / qf / sf / final |
| `result` | TEXT | H / D / A (null until settled) |
| `home_goals` | INT | null until settled |
| `away_goals` | INT | null until settled |

### `features`
| Column | Type | Notes |
|---|---|---|
| `match_id` | TEXT FK → matches.id | |
| `computed_at` | DATETIME | must be < kickoff_utc |
| `elo_delta` | REAL | |
| `form_5_home` | REAL | |
| `form_5_away` | REAL | |
| `pinnacle_implied_home` | REAL | |
| `pinnacle_implied_draw` | REAL | |
| `pinnacle_implied_away` | REAL | |
| `xg_delta_home` | REAL | |
| `xg_delta_away` | REAL | |
| `fifa_rank_delta` | INT | |

### `signals`
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `match_id` | TEXT FK | |
| `market_ticker` | TEXT | Kalshi ticker |
| `side` | TEXT | YES / NO |
| `model_prob` | REAL | |
| `market_implied` | REAL | kalshi YES price |
| `edge` | REAL | model_prob − market_implied |
| `kelly_fraction` | REAL | |
| `bet_size_cents` | INT | dollar amount × 100 |
| `generated_at` | DATETIME | |

### `orders`
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | Kalshi order_id |
| `signal_id` | INT FK → signals.id | |
| `status` | TEXT | pending / filled / canceled |
| `limit_price` | REAL | |
| `contracts` | INT | |
| `filled_price` | REAL | null until filled |
| `placed_at` | DATETIME | |
| `settled_at` | DATETIME | null until match resolves |
| `pnl_cents` | INT | null until settled |

### `bankroll_log`
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `timestamp` | DATETIME | |
| `balance_cents` | INT | |
| `event` | TEXT | deposit / win / loss / fee |

---

## 11. Dashboard (`dashboard/app.py`)

Minimum viable CLI dashboard using the `rich` library:

```
┌─────────────────────────────────────────────────────┐
│  KALSHI WC BOT — Live Dashboard     2026-06-15 18:42 │
├──────────────────┬──────────────────────────────────┤
│ Bankroll         │ $1,247.83  (+$47.83 / +3.99%)   │
│ Open Positions   │ 2                                │
│ Total Bets       │ 14                               │
│ Hit Rate         │ 57.1% (8/14)                     │
│ Max Drawdown     │ -8.2%                            │
├──────────────────┴──────────────────────────────────┤
│ OPEN POSITIONS                                       │
│  KXWC26-ARG-W   YES @ 0.61   $50   current: 0.67   │
│  KXWC26-BRA-GRP YES @ 0.72   $35   current: 0.74   │
├─────────────────────────────────────────────────────┤
│ NEXT SIGNALS (T-2h)  Model  Market  Edge            │
│  France vs Morocco   0.58   0.51    +7.0%  ✓ BET   │
│  Spain vs Portugal   0.44   0.44    +0.0%  ✗ SKIP  │
└─────────────────────────────────────────────────────┘
```

---

## 12. Scheduler (`scheduler/jobs.py`)

Using APScheduler with the following daily jobs:

| Job | Schedule | Action |
|---|---|---|
| `refresh_odds` | Every 6h during WC | Pull The Odds API + Kalshi prices |
| `refresh_injuries` | Daily 8:00 AM CT | Pull API-Football injuries/lineups |
| `generate_signals` | T-2h before each match | Run model + edge detection |
| `place_orders` | T-30min before each match | Execute orders via Kalshi API |
| `settle_positions` | T+2h after each match | Pull result, settle, log P&L |
| `update_bankroll` | After each settlement | Sync with Kalshi balance endpoint |

---

## 13. Config & Secrets

All configuration loaded from `.env` via `python-dotenv`. **Never hardcode.**

```env
# .env.example — copy to .env and fill in values

# API Keys
KALSHI_API_KEY=your_kalshi_api_key
KALSHI_API_SECRET=your_kalshi_api_secret
API_FOOTBALL_KEY=your_api_football_key
THE_ODDS_API_KEY=your_odds_api_key

# Kalshi environment
KALSHI_ENV=prod  # or demo for paper trading

# Strategy parameters
MIN_EDGE_THRESHOLD=0.04
MAX_BET_FRACTION=0.05
MAX_PORTFOLIO_EXPOSURE=0.20
KELLY_FRACTION=0.5
STOP_LOSS_THRESHOLD=0.25

# Database
DB_PATH=data/db.sqlite

# Logging
LOG_LEVEL=INFO
```

---

## 14. Testing Strategy

- **Unit tests:** Every pure function in `features/`, `strategy/`, `backtest/` must have tests
- **Backtester integrity test:** Run 2018 WC backtest; assert zero look-ahead violations
- **Kelly sanity test:** Assert bet size never exceeds `MAX_BET_FRACTION × bankroll`
- **Edge detection test:** Known odds input → known expected edge output
- **Order manager mock test:** Mock Kalshi API; assert correct order params
- Run with: `pytest tests/ -v`

---

## 15. Development Phases

### Phase 1 — Data & Backtester (Weeks 1–3)
- [ ] Set up project structure and virtual environment
- [ ] Build `ingestion/football_data_co.py` — download and parse historical CSVs
- [ ] Build `ingestion/statsbomb.py` — pull 2022 WC event data
- [ ] Build `features/pipeline.py` with ELO, form, odds features
- [ ] Build `backtest/engine.py` with look-ahead guard
- [ ] Validate: run 2018 WC backtest on flat-bet baseline

### Phase 2 — Model (Weeks 4–5)
- [ ] Train logistic regression baseline; record Brier score
- [ ] Train XGBoost model; tune hyperparameters
- [ ] Apply calibration; validate reliability diagram
- [ ] Run 2018 WC backtest with model-driven bets
- [ ] Run 2022 WC holdout (once only)

### Phase 3 — Live Plumbing (Weeks 6–7)
- [ ] Build `ingestion/api_football.py` — live fixtures, injuries
- [ ] Build `ingestion/kalshi.py` — live prices, order book
- [ ] Build `execution/order_manager.py`
- [ ] Paper-trade on Kalshi demo environment (KALSHI_ENV=demo)
- [ ] Build `dashboard/app.py`

### Phase 4 — Live Trading (World Cup, June–July 2026)
- [ ] Switch to `KALSHI_ENV=prod` with small initial bankroll ($100–$200)
- [ ] Monitor daily; adjust `MIN_EDGE_THRESHOLD` if signal count is too high/low
- [ ] Document all deviations from model predictions for post-tournament analysis

---

## 16. Risk Acknowledgments

- Prediction markets involve real financial risk — never bet more than you can afford to lose
- Model probabilities are estimates, not certainties — the 2022 WC had multiple upsets that would have hurt any model
- Kalshi is CFTC-regulated but prediction market liquidity is lower than major sportsbooks — large orders can move the market
- The 2026 WC has 104 matches (expanded 48-team format) vs 64 in 2022 — more games = more opportunities but also more unfamiliar matchups
