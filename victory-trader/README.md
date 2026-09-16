# Victory Trader US

Research codebase for studying short-horizon momentum in low-priced U.S. equities.

## Current scope

- U.S. exchange-listed low-priced equities, with common-stock reference filtering planned next
- Research price universe: prior-session close roughly $0.50 to $20
- 1-minute OHLCV as the first-resolution dataset
- Event-study first: detect large intraday moves, then measure subsequent price paths
- Market-wide daily candidate screening before downloading minute bars
- No live-money execution in this phase

## Principles

1. Separate research from execution.
2. Never commit API keys or credentials.
3. Prefer point-in-time observable features.
4. Treat backtest results as hypotheses until they survive out-of-sample and paper-trading validation.
5. Model transaction costs, slippage, halts, and corporate actions before trusting simulated P&L.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -e .[dev]
copy .env.example .env
```

Put your Massive API key in `.env`, then check connectivity:

```bash
victory-trader check-api
```

Run an event study for one ticker/day:

```bash
victory-trader event-study TICKER 2026-09-15
```

Build a market-wide event dataset for one U.S. trading day:

```bash
victory-trader market-dataset 2026-09-15
```

The default market scan uses:

- prior-session close between $0.50 and $20
- target-session high at least +20% above the prior close
- OTC excluded at the Massive grouped-market request
- conservative 12.5-second spacing between Massive requests for the free-plan workflow
- output to `data/events/market_events_YYYY-MM-DD.parquet`

Useful research controls:

```bash
victory-trader market-dataset 2026-09-15 \
  --min-price 0.50 \
  --max-price 20 \
  --min-high-return 20 \
  --min-dollar-volume 1000000 \
  --max-candidates 10
```

`data/`, `.env`, and artifacts are intentionally excluded from Git.

## Implemented

- Massive REST connectivity
- Historical previous-close lookup
- 1-minute aggregate normalization
- First threshold crossing detection (+10/+20/+30/+50/+75/+100%)
- Post-event returns at 1/2/5/10/15/30/60 minutes
- MFE and MAE measurement using future bars only
- Market-wide grouped-daily candidate screening
- Per-candidate minute-bar event extraction
- Parquet/CSV event dataset output
- Automated pytest suite in GitHub Actions

## Next milestones

- Reference-data filter for point-in-time common stocks only
- Corporate-action and split audit layer
- Multi-day / multi-year dataset builder
- Descriptive event analytics and short-horizon exit comparisons
- Cost-aware backtester
- Walk-forward / out-of-sample validation
- Paper-trading integration
