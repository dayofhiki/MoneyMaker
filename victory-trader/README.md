# MoneyMaker

MoneyMaker is a research platform for testing short-horizon momentum edges in low-priced U.S. equities before any paper or live deployment.

The core design goal is not to manufacture an attractive backtest. It is to make false edges difficult to survive.

The long-term architecture is a stateful trader, not a fixed-horizon
classifier. The first orchestration layer is now implemented in
`victory_trader.attention_runtime`: every broad-market scan can move tickers
through `SCAN -> WATCH -> HOT -> POSITION -> WATCH/DROP`, reallocate finite
high-resolution attention, and request progressively richer observation tiers.
See `research/ROADMAP.md` and
`research/hierarchical-attention-runtime-v0.1.md`. This runtime is research
infrastructure; it is not evidence of a validated edge or permission for live
execution.

## Current formal research scope

- U.S. common stocks primarily listed on Nasdaq, NYSE, or NYSE American
- prior-session nominal/as-traded close from $0.50 to $20
- OTC excluded
- same-day split events excluded
- **regular-session** +10/+20/+30/+50/+75/+100% first close-based threshold crossings
- premarket data retained as point-in-time context, not as a formal v0.2 signal session
- executable primary entry at the exact next-minute open after the signal bar closes, while the market is still in the regular session
- +1m/+2m additional entry-latency sensitivity
- exact-clock 1/2/5/10/15/30/60-minute regular-session outcomes
- explicit light/base/stress spread+slippage assumptions
- gap-aware and regular-close-aware barrier handling
- cluster-aware uncertainty by ticker-day and trading day
- official U.S. equity calendar handling for holidays and early closes
- optional official Nasdaq halt annotations
- resumable historical collection with response cache, daily checkpoints, manifests, and bounded workflow chunks
- no live-money execution in this phase

Formal v0.2 uses regular-session events because the grouped daily discovery layer is suitable as a superset for regular-session threshold events, while market-wide extended-hours event discovery requires a different raw minute/trade acquisition layer. Extended-hours signals are a future protocol version, not silently mixed into v0.2.

## Important research rules

1. Completed daily high may only accelerate historical candidate discovery.
2. Discovery threshold cannot exceed the lowest event threshold being studied. v0.2 therefore uses +10% discovery.
3. Completed-day high/close/volume/dollar-volume never become entry features.
4. Historical nominal prices define the $0.50-$20 universe so future splits do not rewrite universe membership.
5. Provider market-cap/share-count fields are excluded from initial ML features until publication-time semantics are proven safe.
6. A close-based event is not executable at that same close. Primary entry is the next exact minute open.
7. Missing entry/exit minutes remain missing. A later bar is never substituted.
8. TP/SL double-hits inside one minute remain ambiguous.
9. Stop gaps use the first observed open rather than granting the requested stop price.
10. A regular-session barrier position that survives to the close is exited at the last observable regular-session bar; if that bar is unavailable, the outcome is explicitly unresolved.
11. Multiple thresholds on the same ticker-day are clustered, not treated as independent experiments.
12. ML features are an explicit allowlist. New dataframe columns never silently become features.
13. No result is trusted until chronological validation, untouched holdout, and later realtime/finer-data validation.

See `RESEARCH_PROTOCOL.md` for the frozen v0.2 protocol.

## Local quick start

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -e .[dev]
copy .env.example .env
```

Put the Massive API key in `.env`. Never commit it.

```bash
moneymaker check-api
moneymaker event-study TICKER 2026-09-15
moneymaker market-dataset 2026-09-15 --annotate-halts
moneymaker multi-day-dataset 2026-04-01 2026-04-05 --annotate-halts
moneymaker analyze-dataset data/events/market_events_2026-04-01_2026-04-05.parquet --horizon 5
```

The old `victory-trader` CLI alias remains temporarily available for compatibility.

## Data architecture

The expensive acquisition loop is separated from repeated research:

```text
Massive API
   ↓
secret-free HTTP response cache
   ↓
daily resumable checkpoints + manifest
   ↓
normalized event dataset
   ↓
explicit model feature allowlist / outcomes
   ↓
statistics → baseline strategy → ML → validation
```

Historical responses are cached under `data/cache/massive/`. Cache hits return immediately and do not incur API pacing delay. Daily completed states live under `data/checkpoints/` and distinguish event/no-event/closed-market days. Unexpected failures are not marked complete.

The GitHub Actions cache/checkpoint layer is operational persistence, not a permanent market-data warehouse. A durable private object store can replace it later without changing the research schema. Provider data should not be committed to source control merely because the repository is private.

## GitHub Actions research runner

`.github/workflows/victory-trader-research.yml` is displayed as **MoneyMaker Research**.

Configure repository Actions secret `MASSIVE_API_KEY`, then manually run the workflow. A requested date range is divided into bounded calendar chunks, default 5 days. Chunk jobs run sequentially so one API key is not hit concurrently. Each successful chunk saves cache/checkpoint state and uploads its own artifact; a final job merges all current-run chunks and performs aggregate audit/analysis.

For formal research:

- leave `max_candidates` blank,
- keep the conservative request interval unless the provider plan changes,
- use bounded chunks rather than increasing job timeout,
- inspect manifests when counts or exclusions look suspicious.

## Point-in-time features

Current model-eligible feature families include:

- threshold, signal price, prior close
- event and cumulative volume
- true clock-time 5/15/30-minute volume
- volume acceleration versus prior clock-time baseline
- same-time historical RVOL, excluding incompatible early-close dates
- extended-session VWAP distance
- regular-session-anchored VWAP distance
- observed-so-far HOD distance
- exact-clock trailing return
- consecutive-minute trailing volatility
- premarket return/volume
- minutes from official regular open
- common-stock/exchange taxonomy

`MODEL_FEATURE_COLUMNS` is the only supported feature bridge into ML. Future returns, MFE/MAE, halt-after-event labels, ticker identity, completed-day summaries, executable entry price, and unverified historical fundamentals are not model inputs.

## Executable outcome model

The threshold event is known only when its one-minute signal bar has closed. v0.2 therefore records:

- `signal_price`: event-bar close, diagnostic only
- `entry_price`: exact next-minute open, primary
- delay-1/delay-2 entry prices and outcomes for latency sensitivity

Primary research endpoint:

- 5-minute delay-0 **base-friction-adjusted** return

The friction scenarios are:

- light: 10 bps half-spread + 10 bps adverse slippage, minimum half-spread 0.5 cent
- base: 25 bps half-spread + 25 bps adverse slippage, minimum half-spread 1 cent
- stress: 75 bps half-spread + 75 bps adverse slippage, minimum half-spread 2 cents

These are sensitivity assumptions, not historical quote measurements.

## TP/SL diagnostics

Frozen pairs:

- +2 / -1%
- +3 / -2%
- +5 / -3%
- +10 / -5%

Barrier evaluation uses elapsed clock time. Same-minute double touches remain ambiguous. Stop-market gaps are filled at the first observed open if that open has already crossed the stop. Formal v0.2 never lets a regular-session trade quietly drift into after-hours: unresolved positions are closed on the last regular bar when observable, otherwise marked `unresolved_session_close`.

## Analytics

`moneymaker analyze-dataset` reports:

- continuation by threshold
- gross vs light/base/stress friction
- ticker-day and trading-day cluster-bootstrap uncertainty
- lower-tail diagnostics including worst return, 1%/5% quantiles, and 5% CVaR
- ticker-day return concentration
- monthly stability
- prior-close price buckets
- intraday time buckets
- +1m/+2m latency sensitivity
- entry/outcome missingness
- MFE/MAE
- barrier diagnostics including stop gaps, ambiguity, regular-close exits, and unresolved rates
- RVOL buckets

Means and win rates should be read alongside medians, tail losses, sample counts, independent cluster counts, concentration, and chronological stability.

## Formal sequence

1. engineering/debug validation
2. development-set collection in bounded chronological chunks
3. descriptive/cluster-aware statistics
4. simple rule-based baseline
5. point-in-time feature analysis
6. tabular ML only if a stable phenomenon exists
7. frozen chronological validation/walk-forward design
8. untouched final holdout once
9. finer trade/quote or realtime paper validation
10. portfolio simulator and broker mechanics
11. tiny live-money deployment only after all prior gates survive

`No edge found` is a valid successful result. A high win rate is useful only when after-cost expectancy and tail risk are also acceptable.
