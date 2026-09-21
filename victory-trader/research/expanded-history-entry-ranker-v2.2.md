# Expanded-history causal entry ranker v2.2 — pre-registration

## Status

Pre-registered after expanded-history sequential Q v2.1 failed and before
implementing or viewing any v2.2 output.

Use only September 2025 through March 2026 development panels. April 2026 and
later remain sealed. This branch tests one intervention: whether the
previously-positive but sparse within-episode entry-rank signal becomes stable
and sufficiently frequent with more strictly prior training history.

## Motivation fixed before results

State Entry Ranker v0.3 produced positive base-net means in January-March
(+2.008%, +1.985%, +2.027%) but only 2, 11, and 6 trades, below its fixed
15-trade-per-month requirement. Expanded-history v2.1 later showed positive
broad q0 rank correlation but an unstable cost-covering selected tail and
disabled WAIT heads.

This experiment does not tune either failed result. It reuses the exact v0.3
entry gate and model geometry with the new chronological training support
already constructed for v2.1.

## Fixed data and information set

- development panels: 2025-09 through 2026-03;
- existing causal price, volume, breadth, sequence and absolute-price features;
- exact lagged FINRA short-volume, strictly-prior 8-K and publication-safe
  FINRA short-interest features;
- no April 2026 or later data;
- no NBBO, news text, new feature selection, manual interactions, or revised
  cost scenario.

## Frozen model and policy

For 5-, 10-, and 15-minute BUY horizons, retain the v0.3 formulation:

1. fit the existing direct base-net EV regressor;
2. fit a separate regressor to within-ticker-day realized-return percentile
   rank;
3. build rank labels inside training ticker-days only;
4. use the existing three-minute training cadence;
5. derive the rank gate from the chronological calibration partition only.

The executable policy is unchanged:

- direct predicted base-net EV must be at least +0.50%;
- predicted rank must clear the calibration-period 75th percentile gate;
- choose the highest predicted-rank eligible horizon at the current state;
- enter at the first chronological qualifying state;
- at most one BUY attempt per ticker-day;
- an unfillable or unevaluable first signal consumes the attempt.

No threshold, horizon, capacity, feature subset, cost, or fall-through rule may
be changed after results.

## Strict past-only folds

- January 2026: train/calibrate on 2025-09 through 2025-12.
- February 2026: train/calibrate on 2025-09 through 2026-01.
- March 2026: train/calibrate on 2025-09 through 2026-02.

Every fold must prove maximum training day is earlier than minimum evaluation
day.

## Required diagnostics

Report monthly:

- attempted, evaluated and unevaluable BUY counts with mutually exclusive
  reasons;
- gross, base and stress return distributions;
- day-balanced base mean, p05, severe-loss rate and worst day;
- direct-EV and within-episode rank correlations for every horizon;
- gate values learned from calibration data;
- common-episode timing and return deltas versus the frozen earliest-eligible
  comparator;
- external-data coverage;
- position-ledger base/light/stress results, delayed exits, unresolved capital,
  cash blocks and complete-accounting status.

## Development promotion rule

Do not access a fresh validation month unless all conditions hold:

1. at least 15 evaluated trades in each of January, February and March;
2. arithmetic and day-balanced base-net means are positive in every month;
3. day-balanced base-net mean exceeds earliest eligible entry in every month;
4. base p05 and stress mean are not worse than the comparator in any month;
5. within-episode rank correlation is positive for every month and horizon;
6. pooled trading-day-cluster bootstrap 95% lower bound is above zero;
7. required external-data coverage meets the v2.1 thresholds;
8. base-scenario marked account return is positive and
   complete_accounting=true in every month.

A sparse positive result fails. A light-cost profit cannot substitute for the
base scenario. Missing execution references cannot be assigned zero P&L.

## Relation to the trader goal

This branch addresses entry actionability only. It does not claim to implement
the final continuous trader. If and only if this entry policy passes, freeze it
and pre-register a position-state HOLD/SELL model that includes entry price,
unrealized P&L, time in position, available cash and the same causal market
state. Adding SELL/HOLD before entry edge replication would increase
data-mining freedom around a negative entry process.
