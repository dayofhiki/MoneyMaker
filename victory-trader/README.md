# Victory Trader US

Research codebase for studying short-horizon momentum in low-priced U.S. equities.

## Current scope

- U.S. common stocks primarily listed on Nasdaq, NYSE, or NYSE American
- Research price universe: prior-session close roughly $0.50 to $20
- 1-minute OHLCV/VWAP as the first-resolution dataset
- Event-study first: detect large intraday moves, then measure subsequent price paths
- Market-wide daily candidate screening before downloading minute bars
- Same-day stock splits excluded from the research sample
- Point-in-time ticker metadata used for security-type validation
- Point-in-time momentum features plus historical same-time RVOL
- Official Nasdaq halt annotations when the public feed is available
- Explicit light/base/stress execution-friction scenarios
- Persistent Massive response cache for resumable historical research
- No live-money execution in this phase

## Principles

1. Separate research from execution.
2. Never commit API keys or credentials.
3. Prefer point-in-time observable features.
4. Treat backtest results as hypotheses until they survive out-of-sample and paper-trading validation.
5. Model transaction costs, slippage, halts, and corporate actions before trusting simulated P&L.
6. Mark 1-minute TP/SL double-hits as ambiguous instead of assuming favorable ordering.
7. Distinguish missing halt data from a verified absence of halts.
8. Production research uses the full qualifying candidate universe; debug limits must not rank candidates by eventual return.
9. Unexpected API or parsing errors fail the research run instead of silently producing an incomplete dataset.

## Local quick start

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

Build one market-wide event dataset:

```bash
victory-trader market-dataset 2026-09-15 --annotate-halts
```

Build a date-range dataset:

```bash
victory-trader multi-day-dataset 2026-09-01 2026-09-15 --annotate-halts
```

Analyze an accumulated dataset:

```bash
victory-trader analyze-dataset data/events/market_events_2026-09-01_2026-09-15.parquet --horizon 5
```

Historical Massive responses are cached under `data/cache/massive/`, which is ignored by Git. Successful cached requests return immediately on later runs and the cache key never includes the API secret.

## GitHub Actions research runner

The repository includes `.github/workflows/victory-trader-research.yml` for manual cloud runs. Configure a repository Actions secret named `MASSIVE_API_KEY`, then open GitHub Actions -> Victory Trader Research -> Run workflow.

For real research, leave **max_candidates blank**. The workflow restores/saves the Massive cache between runs, so overlapping or restarted historical jobs can reuse previously downloaded responses.

`max_candidates` exists only for smoke/debug runs. When supplied, candidates are selected by a stable hash of trading date + ticker. The limit never uses final day-high return, close, volume, or forward performance.

Each run:

1. checks out the repository,
2. restores the historical API cache,
3. validates the API secret,
4. installs Victory Trader,
5. runs the test suite,
6. builds a multi-day Parquet dataset,
7. runs the descriptive analyzer,
8. uploads the dataset, build log, and analysis text as a GitHub artifact.

This allows research jobs to run without keeping a local machine or terminal open.

## Default market scan

- prior-session close between $0.50 and $20
- target-session high at least +20% above the prior close, used only to discover historical event days efficiently
- OTC excluded at the Massive grouped-market request
- common-stock type (`CS`) only
- primary exchange limited to XNAS / XNYS / XASE
- same-day split events excluded
- conservative 12.5-second spacing between uncached Massive requests for the free-plan workflow
- Parquet output under `data/events/`

Each candidate's minute request covers roughly 35 calendar days through the target day. That single ranged request provides both target-day bars and prior intraday context for RVOL, avoiding an extra candidate-level API request.

## Point-in-time event features

Features are calculated only from information available at or before each threshold crossing:

- event / 5m / 15m / 30m volume
- cumulative volume
- 1m and 5m volume acceleration versus the prior 20 minutes
- cumulative and 5m RVOL versus up to 20 prior trading dates at the same clock time
- session VWAP and event distance from VWAP
- event distance from high-of-day observed so far
- trailing 5m / 15m / 30m return and volatility
- premarket return and volume
- minutes from the regular-session open
- premarket / regular / after-hours session flags

Trailing returns are stored under `trailing_return_*` names so they cannot collide with forward outcome columns such as `return_5m_pct`.

## Short-exit experiments

Every event receives first-hit labels for these initial TP/SL pairs over the research horizon:

- +2% / -1%
- +3% / -2%
- +5% / -3%
- +10% / -5%

A bar that touches both TP and SL is marked `ambiguous`, because 1-minute OHLC cannot reveal which happened first.

Forward-return horizons use actual timestamps rather than simply counting bars. If the exact target minute is missing, for example around a halt or data gap, that horizon is left missing instead of substituting a later bar.

## Halt annotations

With `--annotate-halts`, the builder queries Nasdaq Trader's official historical trade-halt feed and records whether a halt begins within 5/15/30/60 minutes after each event, the first halt reason, estimated halt duration when a resumption trade time is available, and whether the reason is a volatility-pause code such as LUDP/LUDS.

Halt feed failure is recorded as `halt_data_available = False`; it is not interpreted as a no-halt observation.

## Execution-friction scenarios

Minute bars do not reveal the exact historical bid/ask spread or market impact. Instead of pretending otherwise, each outcome is recalculated under explicit sensitivity scenarios:

- `light`: 10 bps half-spread + 10 bps adverse slippage, minimum half-spread 0.5 cent
- `base`: 25 bps half-spread + 25 bps adverse slippage, minimum half-spread 1 cent
- `stress`: 75 bps half-spread + 75 bps adverse slippage, minimum half-spread 2 cents

These are research stress assumptions, not claims about actual broker fills. The minimum-cent component matters especially for $0.50-$5 stocks.

## Implemented

- Massive REST connectivity with Authorization-header authentication
- Persistent secret-free historical response cache
- Historical previous-close lookup
- Point-in-time ticker metadata lookup
- Same-day split lookup
- 1-minute aggregate normalization including optional VWAP/trade count
- First threshold crossing detection (+10/+20/+30/+50/+75/+100%)
- Exact-clock post-event returns at 1/2/5/10/15/30/60 minutes
- MFE and MAE measurement using future bars only
- Point-in-time momentum feature extraction
- Same-time historical RVOL context
- TP/SL first-hit labels with conservative ambiguous handling
- Market-wide grouped-daily candidate screening
- Full-universe research mode plus outcome-independent debug subsampling
- Common-stock / exchange universe filtering
- Split-day exclusion
- Optional official Nasdaq halt annotation
- Light/base/stress execution-friction sensitivity model
- Single-day and multi-day Parquet/CSV dataset output
- Fail-fast protection against silent partial multi-day datasets
- Dataset analytics by threshold, execution-cost scenario, RVOL bucket, MFE/MAE, and TP/SL experiment
- Automated pytest suite in GitHub Actions
- Manual GitHub Actions research runner with downloadable artifacts and cross-run cache reuse

## Next milestones

- Run the first full-universe multi-week sample and inspect descriptive statistics
- Freeze an initial Momentum Hunter hypothesis before parameter exploration
- Add float / point-in-time share-count quality checks and float-turnover experiments
- Add finer quote or trade data where 1-minute ambiguity materially affects results
- Refine spread/slippage assumptions using observable market-quality data
- Add partial-fill and halt-aware execution simulation
- Build train / validation / untouched holdout partitions
- Walk-forward / out-of-sample validation
- Paper-trading integration
