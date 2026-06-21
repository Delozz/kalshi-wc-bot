# Demo Paper Run

The bot's full live loop can be run in a single observable pass before any real money
is at stake. This is the L8 demo-first checkpoint: validate on the Kalshi **demo**
environment before ever touching prod.

## One cycle, dry-run (no orders sent)

```powershell
python -m scheduler.jobs --once
```

Runs, in order: refresh market data → sync portfolio → generate signals → settle
finished matches → update bankroll → render the dashboard. Without API credentials it
degrades gracefully (logs what is missing), so this is useful just to verify the wiring.

## One cycle placing REAL demo orders

1. Create a Kalshi **demo** account and API key; download the RSA private key.
2. Copy `.env.example` to `.env` and fill in:

   ```env
   KALSHI_ENV=demo
   KALSHI_API_KEY=<demo key id>
   KALSHI_API_SECRET=<path to the demo private-key .pem, or the PEM text itself>
   API_FOOTBALL_KEY=<key>     # live WC fixtures (free tier 100/day)
   THE_ODDS_API_KEY=<key>     # optional no-vig cross-check (free tier 25/day)
   ```

3. Train the model so an artifact exists in `model/artifacts/`:

   ```powershell
   python -m model.train
   ```

4. Run one cycle that places **demo** orders:

   ```powershell
   python -m scheduler.jobs --once --live-orders
   ```

## Continuous (live tournament)

```powershell
python -m scheduler.jobs        # blocking scheduler; signals are dry-run by default
```

## Safety (L8)

- The default is **dry-run**; `--live-orders` is required to place anything.
- Prod (`KALSHI_ENV=prod`) additionally requires `KALSHI_ALLOW_PROD_ORDERS=1`, and only
  after a clean demo paper run with no unexpected fills.

## Verify before trusting it

- **Market resolver:** `strategy/signal_gen.default_market_resolver` matches a fixture
  to a Kalshi market by best-effort title text. Confirm it against the real `KXWC26`
  market structure (ticker format; 3-way vs per-outcome).
- **Team aliases:** `features/teams.py` maps source names to martj42 canonical names.
  Confirm against real API-Football fixture names.
- v1 trades the **home-win YES** contract only (draw/away not yet implemented).
- The model's real edge is unproven until this live forward test — there is no
  historical WC odds data to backtest ROI against.
