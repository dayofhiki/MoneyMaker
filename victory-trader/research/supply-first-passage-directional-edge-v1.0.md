# Request 201 — point-in-time supply-turnover first-passage edge

## Purpose

Requests 199 and 200 show that the current regular-session state and the frozen
premarket context do not create a useful directional selected tail.

Request 201 tests a different causal hypothesis:

**the same observed buying pressure can imply different continuation odds when
the point-in-time share supply differs by orders of magnitude.**

An earlier outcome-blind supply probe established that Massive point-in-time
ticker details queried at D-1 calendar day have high strictly-prior coverage.
An older fixed-15m supply branch preserved broad ordering and improved some
months but did not produce a robust profitable policy. This request does not
reuse its action rule. It asks only whether supply/turnover context improves
the cleaner first-passage directional target.

## Data

No new dates.

- train: Request-171 FIT only;
- evaluate: Request-171 chronological CALIBRATION only;
- Request-178 remains sealed.

Execution / label contract is exactly Request 199:

- BASE friction;
- 1-second entry latency;
- 5-second entry expiry;
- -3% stop;
- tight +5% upside;
- runner +10% upside;
- 30-minute first-passage observation cap.

## Strict-prior supply source

For every unique candidate ticker-day D, query:

    GET /v3/reference/tickers/{ticker}?date=D-1 calendar day

Rules:
- query date must be strictly earlier than D;
- never query D or a later day;
- never fall back to current ticker details;
- missing point-in-time fields remain missing.

Raw fields used:
- weighted_shares_outstanding;
- share_class_shares_outstanding.

Provider market cap may be retained as a diagnostic but is not a model feature.

## Frozen supply-turnover features

Append exactly these eight causal features to the Request-199 no-rich-second
comparator:

1. supply_log_weighted_shares_prior;
2. supply_log_share_class_shares_prior;
3. supply_log_implied_market_cap_prior;
4. supply_minute_weighted_turnover;
5. supply_minute_share_class_turnover;
6. supply_regular_cum_weighted_turnover;
7. supply_regular_cum_share_class_turnover;
8. supply_share_class_to_weighted_ratio.

Definitions:

- implied market cap = weighted shares * strictly causal previous close;
- current-minute volume comes from the completed state;
- regular cumulative volume sums one-second aggregate volume from the official
  regular-session open up to but not including state_t;
- all turnover ratios are share volume divided by the respective strictly-prior
  share count.

No low-float threshold, market-cap threshold, turnover cutoff, interaction, or
feature selection is chosen from calibration outcomes.

## Models

For each barrier pair train:

1. comparator: exact Request-199 no-rich-second feature family;
2. supply: comparator plus the eight frozen supply-turnover features.

Use the same equal-day HGB classifier family as Requests 199/200. No calibration
offset or action threshold is fitted.

## Required report

Report:
- request success;
- weighted-share coverage;
- share-class coverage;
- strict-prior query check;
- comparator and supply AUC / AP;
- AUC uplift;
- decisive-only AUC / AP;
- per-day AUC and supply-vs-comparator day wins;
- P80/P90/P95 supply-score barrier diagnostics;
- all-state and P90 day-balanced barrier proxy.

## Development signal rule

A supply branch is worth converting into an executable action model only if at
least one barrier pair satisfies all:

1. supply AUC >= 0.62;
2. AUC uplift over comparator >= +0.03;
3. supply AUC exceeds comparator on at least 6 of 8 calibration days;
4. P90 day-balanced barrier proxy improves by at least +0.50 percentage points
   versus all states;
5. weighted-share coverage >= 80% and every populated query date is strictly
   prior.

This is not a promotion rule. Request-178 remains sealed.

## Failure rule

If Request 201 fails, do not tune float cutoffs, turnover thresholds, share
definitions, D-1 query timing, barrier sizes, score quantiles or model capacity
on calibration outcomes.

Move to a new causal information source rather than recombining two failed
context branches.
