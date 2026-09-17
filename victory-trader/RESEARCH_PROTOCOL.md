# Victory Trader US — Research Protocol v0.1

Status: **frozen before full-universe historical research**

This document exists to reduce researcher degrees of freedom. Changes are allowed later, but every change must be versioned and must not retroactively redefine the untouched final holdout.

## 1. Research objective

Test whether low-priced U.S. common stocks that have already entered an extreme intraday momentum state contain short-lived, repeatable continuation that survives explicit execution-friction assumptions.

The system is not trying to predict the day's ultimate winner. The core question is whether a mechanical strategy can harvest short continuation segments after an observable threshold crossing while limiting time exposed to reversals, halts, and microstructure costs.

## 2. Universe

Initial universe:

- U.S. stocks market
- common-stock type (`CS`)
- primary exchange XNAS / XNYS / XASE
- prior-session close from $0.50 through $20.00
- OTC excluded
- same-day split events excluded

The completed daily high may be used to efficiently discover historical dates/tickers that contained a threshold event. It must not be used as a model feature or to rank/truncate the production research sample.

## 3. Event definition

First close-based crossing of each threshold during a trading date:

- +10%
- +20%
- +30%
- +50%
- +75%
- +100%

Reference price is the prior-session close.

An event is observable only after the crossing minute bar closes. All model features must use information at or before that timestamp.

## 4. Primary outcomes

Forward returns at exact clock-time horizons:

- 1 minute
- 2 minutes
- 5 minutes
- 10 minutes
- 15 minutes
- 30 minutes
- 60 minutes

If the exact future minute is unavailable, the outcome is missing. A later bar must not be substituted.

Primary short-horizon research endpoint for v0.1:

- **5-minute base-friction-adjusted return**

Supporting outcomes:

- gross 5-minute return
- 1/2/10/15/30/60-minute returns
- MFE / MAE
- positive-return rate
- median return

## 5. Pre-specified short-exit experiments

The initial barrier family is frozen as:

- TP +2% / SL -1%
- TP +3% / SL -2%
- TP +5% / SL -3%
- TP +10% / SL -5%

A 1-minute bar touching both TP and SL is `ambiguous` and is not assigned a favorable sequence.

These four pairs are a diagnostic family, not permission to endlessly search arbitrary TP/SL combinations. Any expanded grid requires a new protocol version and validation discipline.

## 6. Execution-friction sensitivity

The v0.1 friction scenarios are frozen as:

- light: 10 bps half-spread + 10 bps slippage, minimum half-spread 0.5 cent
- base: 25 bps half-spread + 25 bps slippage, minimum half-spread 1 cent
- stress: 75 bps half-spread + 75 bps slippage, minimum half-spread 2 cents

The **base** scenario is the primary v0.1 research scenario. Light and stress are sensitivity checks.

These are assumptions, not claims about actual fills.

## 7. Point-in-time feature families

Allowed initial feature families:

- threshold level
- event price / prior close
- session label and minutes from regular open
- cumulative and short-window volume
- volume acceleration
- historical same-time RVOL
- session VWAP distance
- distance from high-of-day observed so far
- trailing 5/15/30-minute return
- trailing 5/15/30-minute volatility
- premarket return and volume
- point-in-time market cap / shares outstanding when available
- official halt annotations for outcome/risk analysis

No feature may use the completed day's close/high/volume, future halt information, or future bars as an input to the entry decision.

## 8. Data partitions

The September 2026 sample already used during engineering is designated **debug-only** and is excluded from formal validation claims.

Subject to provider historical availability, the initial frozen chronological partitions are:

- Development / exploratory research: **2024-10-01 through 2026-04-30**
- Validation / walk-forward design: **2026-05-01 through 2026-06-30**
- Final untouched holdout: **2026-07-01 through 2026-08-31**
- Engineering/debug sample: **2026-09-01 onward**, not used as an untouched test

The final holdout must not be queried for strategy selection, parameter tuning, feature selection, or threshold selection until a candidate strategy and evaluation procedure are frozen.

## 9. First full-universe research sequence

Before touching validation or holdout data:

1. Run a full-universe two-week development slice to validate data quality and event counts.
2. Expand development data in chronological chunks.
3. Produce descriptive statistics without selecting a final strategy from one best-looking cell.
4. Look for broad, stable neighborhoods rather than a single optimum.
5. Freeze a candidate rule/model and evaluation procedure.
6. Run chronological validation / walk-forward analysis.
7. Only after that, evaluate the untouched final holdout once.

## 10. Required integrity gates

A production research dataset must pass the automated dataset audit:

- no duplicate event keys
- no debug candidate limit
- required forward horizons present
- gross and friction-adjusted missingness aligned
- trailing-return features separated from future-return outcomes
- positive entry/reference prices
- unexpected API/code failures must stop the run rather than silently skip data

## 11. Interpretation rules

- Means must be accompanied by medians and sample sizes.
- Win rate alone is insufficient.
- Concentration in a few tickers/days must be examined.
- High-threshold events with few observations are not treated as established effects.
- Results must be checked across price buckets, time-of-day, market-cap/liquidity regimes, and time periods before being called robust.
- A result that disappears under modest friction or neighboring parameters is not considered a durable edge.
- `No edge found` is a valid research outcome.
