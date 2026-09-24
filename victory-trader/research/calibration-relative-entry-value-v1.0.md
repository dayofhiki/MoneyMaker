# Request 183 — calibration-relative entry-value support diagnostic

## Status

Pre-registered after Request 182 produced positive fresh ranking for both
ENTER_NOW and WAIT_1M value heads but selected zero trades at the absolute
predicted-value > 0 boundary.

This request opens no new dates and changes no upstream policy. The already
opened June 23/24/25/26/29 block is used only to decide whether relative
entry-value support exists before investing in a richer entry representation.

## Hypothesis

Absolute BASE-value calibration can shift across regimes even when ordering
survives. A threshold fixed from the old calibration score distribution may
retain only historically exceptional opportunities without using any fresh
outcomes.

## Frozen models and data

Reproduce Request 182 exactly:

- same BUY-gated historical fit/calibration POSITION population;
- same causal HOT-time minute/attention features;
- same ENTER_NOW and WAIT_1M targets;
- same HGB regressors, seeds, day weights, winsorization and residual offsets.

For each calibration row compute:

    best_predicted_value = max(V_enter, V_wait)

Freeze the 80th, 90th and 95th percentiles of that calibration distribution.

## Fresh candidate policies

For each fresh BUY opportunity:

1. choose ENTER_NOW or WAIT_1M solely by which predicted value is larger;
2. execute that action only if its best predicted value is at or above one of
   the frozen calibration percentile thresholds;
3. otherwise ABSTAIN.

No fresh outcome may alter the thresholds.

Evaluate three preregistered support levels independently:

- CAL_P80;
- CAL_P90;
- CAL_P95.

## Required metrics

For each support level report:

- frozen score threshold;
- executed rate and action split;
- executed BASE mean and day-balanced mean;
- all-opportunity value including abstentions;
- positive-trade and severe-loss rates;
- matched improvement over current ENTER_NOW;
- 10,000-sample trading-day bootstrap of the matched improvement;
- number of positive-return trading days.

Also report the two action-head fresh Spearman values.

## Development support gate

A support level is considered a development pass only if:

1. action-value coverage >=90%;
2. executed rate is between 3% and 40%;
3. executed-trade day-balanced BASE mean >0;
4. all-opportunity day-balanced value >0;
5. matched policy-minus-current day-balanced difference >0;
6. matched-difference bootstrap 95% lower bound >0;
7. severe-loss rate is no worse than current ENTER_NOW;
8. executed trade return is positive on at least 3/5 days;
9. both action heads retain positive fresh Spearman.

If multiple levels pass, choose the least selective passing level
(P80 before P90 before P95) for any later untouched validation.

Request 183 is development-only and never promotion-eligible.
