# Victory Trader US

Research codebase for studying short-horizon momentum in low-priced U.S. equities.

## Current scope

- U.S. exchange-listed common stocks
- Research price universe: roughly $0.50 to $20
- 1-minute OHLCV as the first-resolution dataset
- Event-study first: detect large intraday moves, then measure subsequent price paths
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

Put your Massive API key in `.env`, then run:

```bash
python -m victory_trader.cli check-api
```

## Planned milestones

- Phase 1: API connectivity + local parquet cache
- Phase 2: intraday event detector
- Phase 3: event-study analytics
- Phase 4: cost-aware backtester
- Phase 5: walk-forward / out-of-sample validation
- Phase 6: paper-trading integration
