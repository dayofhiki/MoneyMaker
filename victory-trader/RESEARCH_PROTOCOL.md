# MoneyMaker — Research Protocol v0.2

Status: **frozen before formal development-set collection**

This document limits researcher degrees of freedom. Any material change after formal collection begins requires a new protocol version. The untouched final holdout may never be redefined in response to its observed result.

## 1. Research objective

Test whether low-priced U.S. common stocks that have already entered an extreme **regular-session** intraday momentum state contain short-lived, repeatable continuation that survives executable-entry assumptions, transaction-cost stress, clustering, and chronological out-of-sample validation.

The first goal is not a high backtest win rate. The goal is positive expectancy after realistic frictions with controlled downside tails. Win rate is a supporting diagnostic only.

## 2. Universe

Initial universe:

- U.S. stocks market
- common-stock type (`CS`)
- primary exchange XNAS / XNYS / XASE
- prior-session nominal/as-traded close from $0.50 through $20.00
- OTC excluded
- same-day split events excluded

Universe selection uses unadjusted historical prices so later splits/reverse splits cannot retroactively change whether a stock belonged to the $0.50-$20 universe.

Provider market-cap/share-count fields are excluded from the initial model allowlist until their publication-time semantics are verified. Identity/taxonomy metadata may still be used for common-stock/exchange validation.

## 3. Historical candidate discovery and session scope

The completed daily high may be used only to avoid downloading minute data for ticker-days that could not possibly contain a studied **regular-session** threshold event.

The discovery threshold must be less than or equal to the lowest event threshold under study. With the v0.2 event family, discovery therefore defaults to +10%.

Formal v0.2 event signals are restricted to the official regular session. This is deliberate: Massive minute aggregates include extended-hours activity, while daily OHLC follows the daily consolidated aggregation convention. A market-wide daily-high pre-screen therefore cannot be assumed to be a complete discovery mechanism for premarket-only or after-hours-only threshold events. Premarket bars remain available as point-in-time context features.

Extended-hours event discovery is deferred to a later protocol version using a market-wide minute/trade source capable of discovering those events without future-selected sampling.

Completed daily high/close/volume/dollar-volume:

- may not rank or truncate the production sample,
- may not be model features,
- may not be production entry filters,
- are not emitted into the model-facing event dataset.

A debug candidate limit, when explicitly requested, uses deterministic date+ticker hashing only.

## 4. Event definition

First close-based crossing of each threshold during the official regular session:

- +10%
- +20%
- +30%
- +50%
- +75%
- +100%

Reference price is the prior-session nominal close.

The event becomes observable only after the crossing minute closes. Premarket activity may influence point-in-time features, but a premarket-only threshold crossing is not a formal v0.2 event.

## 5. Executable entry model

Primary v0.2 entry:

- signal: regular-session crossing minute close
- primary entry: exact next minute bar open
- primary entry must itself still be inside the official regular session
- if the exact next-minute bar is absent or lies outside regular hours, the trade is marked not executable and primary outcomes are missing

Latency sensitivity is pre-specified:

- delay 0: next-minute open, primary
- delay 1: one additional clock minute later
- delay 2: two additional clock minutes later

The same regular-session requirement applies to delayed entries. The signal-close return is retained only as an optimistic diagnostic benchmark and is not the primary strategy return.

## 6. Primary outcomes

Executable forward returns at exact holding horizons:

- 1 minute
- 2 minutes
- 5 minutes
- 10 minutes
- 15 minutes
- 30 minutes
- 60 minutes

Primary v0.2 endpoint:

- **5-minute, delay-0, base-friction-adjusted return**

If the exact entry or exit minute required by the model is unavailable, the executable outcome is missing. A later bar is never silently substituted. Formal v0.2 horizon outcomes must remain inside the official regular session; a requested holding horizon that would end after the close is left missing rather than silently switching to after-hours execution.

Supporting outcomes include gross executable return, signal-close benchmark return, MFE/MAE, median, positive-return rate, latency sensitivity, missingness diagnostics, and tail-risk diagnostics.

## 7. Short-exit experiments

Frozen diagnostic barrier family:

- TP +2% / SL -1%
- TP +3% / SL -2%
- TP +5% / SL -3%
- TP +10% / SL -5%

Rules:

- barriers begin only after the executable entry exists,
- elapsed clock time determines the timeout window,
- a bar touching TP and SL is `ambiguous`,
- ambiguous rows are never assigned a favorable ordering,
- a stop-market gap through the stop exits at the first observed bar open rather than the requested stop price,
- formal v0.2 barrier positions never drift silently into after-hours,
- if the requested barrier horizon reaches the regular-session close first, the position is closed at the last observable regular-session bar close,
- if that required regular-session close bar is unavailable, the result is `unresolved_session_close`,
- if an exact non-session-close timeout minute is not tradable/observable, the timeout outcome is unresolved rather than fabricated from an earlier close.

The unresolved/ambiguous rates are themselves reported. Before any barrier strategy is treated as executable, halt/resume and finer trade/quote data must be used to quantify unresolved tail outcomes.

Any expanded TP/SL grid requires a new protocol version.

## 8. Execution-friction sensitivity

Frozen scenarios:

- light: 10 bps half-spread + 10 bps adverse slippage, minimum half-spread 0.5 cent
- base: 25 bps half-spread + 25 bps adverse slippage, minimum half-spread 1 cent
- stress: 75 bps half-spread + 75 bps adverse slippage, minimum half-spread 2 cents

The base scenario is primary. These are sensitivity assumptions, not claims about historical quoted spreads or broker fills.

A candidate edge must not depend on the light scenario alone. Quote/BBO or tick-level validation is required before live deployment.

## 9. Point-in-time feature rules

All model features are explicit opt-ins in `MODEL_FEATURE_COLUMNS`. Newly added dataset columns do not automatically become model inputs.

Initial allowed families:

- threshold level
- signal price and prior close
- minutes from official regular open
- event/cumulative/clock-window volume
- clock-window volume acceleration
- historical same-time RVOL
- extended-session VWAP distance
- regular-session-anchored VWAP distance
- distance from high-of-day observed so far
- exact-clock trailing 5/15/30-minute return
- clock-window trailing volatility using consecutive minutes only
- premarket return and volume
- common-stock/exchange taxonomy

Historical same-time RVOL excludes prior dates where the comparison clock time was outside the regular session because of an early close.

Forbidden model inputs include:

- completed-day high/close/volume/dollar-volume
- future returns, MFE/MAE, TP/SL outcomes
- executable entry price when scoring is intended before that entry
- future halt information
- ticker identity as an initial predictive feature
- provider market cap/share-count fields until publication-time safety is established

## 10. Sessions and exchange calendar

Official U.S. equity exchange-calendar data is used for holidays, previous trading-day lookup, regular-session boundaries, and early closes.

Formal v0.2 signals, entries, and modeled holding horizons are regular-session only. Premarket data is retained as context. Premarket/after-hours event trading is a separate future research problem because it requires an unbiased market-wide extended-hours discovery source and different microstructure assumptions.

## 11. Missing bars and finalized historical aggregates

A missing one-minute aggregate does not mean that a later bar may be substituted. Clock-time features and outcomes preserve the requested wall-clock window.

Historical finalized one-minute bars may contain late/corrected trades that were not present in exactly the same form in real time. Historical 1-minute research is therefore a discovery/validation layer, not the final proof of live reproducibility.

Any strategy that survives historical validation must subsequently survive raw trade/quote or realtime paper-trading validation.

## 12. Data partitions

September 2026 data already inspected during engineering remains debug-only.

Initial chronological partitions:

- Development / exploratory: **2024-10-01 through 2026-04-30**
- Validation / walk-forward design: **2026-05-01 through 2026-06-30**
- Final untouched holdout: **2026-07-01 through 2026-08-31**
- Engineering/debug: **2026-09-01 onward**

Before querying validation/holdout for strategy selection, development data must show adequate independent ticker-day/event counts and market-regime coverage. If the validation/holdout windows are changed for sample-size reasons, the rule must be changed before either period is inspected.

## 13. Statistical dependence and uncertainty

Multiple threshold rows from one ticker-day are not independent experiments.

Required reporting includes:

- raw event count
- unique ticker-day cluster count
- unique trading-day count
- mean and median
- positive-return rate
- 1%/5% lower-tail returns and 5% CVaR for the primary endpoint
- ticker-day cluster bootstrap interval
- trading-day cluster bootstrap interval
- concentration by ticker-day
- monthly, price-bucket, and intraday-time stability

A large raw row count cannot substitute for independent-cluster coverage.

## 14. Multiple testing discipline

Research must prefer broad stable neighborhoods to isolated best cells.

No result is called robust merely because one combination of threshold, RVOL bucket, price bucket, time-of-day, or TP/SL looks strong. Candidate selection must account for parameter sensitivity, neighboring specifications, time stability, and the number of exploratory cuts performed.

Once a candidate rule/model and evaluation procedure are frozen, validation and final holdout are chronological and untouched by further tuning.

## 15. Data acquisition architecture

Massive API responses are an expensive acquisition layer, not the research loop.

Collection rules:

- successful historical responses are cached without credentials,
- cache hits do not incur API pacing sleeps,
- 429 and transient 5xx/network failures use bounded retries/backoff,
- authentication failures remain fatal,
- official calendar lookup avoids wasting provider requests on weekends/holidays,
- collection is split into bounded chronological chunks,
- daily checkpoints and manifests distinguish complete/no-event/closed/error states,
- a completed checkpoint is reused instead of re-downloaded,
- unexpected failures do not silently turn a partial dataset into a valid one.

Research calculations, strategy changes, and ML experiments operate on already-collected event data whenever possible. The HTTP cache/checkpoint layer is operational persistence, not the long-term archival source of truth; a permanent private object store can replace it later without changing the research schema.

## 16. Required dataset integrity gates

A formal dataset must pass automated audit checks including:

- no duplicate event keys
- no debug candidate limit
- nominal/unadjusted universe provenance
- discovery threshold no higher than the lowest event threshold
- regular-session signal scope
- regular-session executable entry requirement
- no completed-day/fundamental leakage columns
- next-minute-open primary entry provenance
- required exact-clock horizons
- gross/net missingness alignment
- no executable outcome without executable entry
- latency sensitivity fields present
- explicit model allowlist complete and future-safe
- non-overlapping session labels
- positive observable prices
- unresolved barrier outcomes explicitly reported rather than dropped silently

## 17. Research sequence

1. Finish engineering/debug tests.
2. Run bounded full-universe development chunks and inspect manifests/data quality.
3. Accumulate development history without inspecting validation/holdout.
4. Produce descriptive and cluster-aware statistics.
5. Establish a simple rule-based baseline.
6. Investigate point-in-time feature predictive power and stable interactions.
7. Introduce tabular ML only after a baseline phenomenon exists.
8. Freeze candidate strategy/model and evaluation procedure.
9. Perform chronological validation/walk-forward analysis.
10. Evaluate final untouched holdout once.
11. Validate finalists on finer trade/quote data and realtime paper trading.
12. Build portfolio/broker mechanics before tiny live-money deployment.

## 18. ML role

Initial ML is an event-quality/edge estimator, not a universal stock-price oracle.

Examples:

- expected 5-minute base-friction-adjusted return after an observable event,
- probability a defined TP is reached before a defined SL,
- ranking simultaneous executable signals.

Initial model families should be simple and auditable: regularized linear/logistic models and tree/boosting models. Random train/test splits are not permitted. Deep sequence models are deferred until finer trade/quote sequences justify them.

Risk and execution controls always outrank an ML score.

## 19. Portfolio/live requirements after an event-level edge exists

Before paper/live claims, add:

- simultaneous-signal allocation
- position sizing
- rules for later thresholds after an existing position
- max positions/trades per day
- daily loss stop and kill switch
- capital/buying-power/settlement constraints
- partial fills and order rejection
- broker/API latency and reconnect/reconciliation behavior
- halt/resume handling

Event-study expectancy alone is not a tradable portfolio backtest.

## 20. Interpretation

`No edge found` is a valid successful research result.

A candidate edge is credible only if it remains positive after primary friction, does not rely on a few ticker-days, is stable across nearby specifications and time segments, survives chronological validation/holdout, and later reproduces under finer execution data or realtime paper trading.

A high win rate is never accepted as a substitute for positive expectancy and controlled tail risk. MoneyMaker may optimize win rate only subject to those constraints.
