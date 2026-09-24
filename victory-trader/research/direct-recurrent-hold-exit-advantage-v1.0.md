# Request 179 — direct recurrent HOLD/EXIT advantage diagnostic

## Status

Pre-registered after Request 178 failed its fresh promotion gate and before
running Request 179.

Request 178 showed a specific split: post-entry value ranking survived on the
fresh June 23/24/25/26/29 block, but the absolute consensus-EV zero boundary
did not transfer. Request 179 therefore changes the target from oracle-style
remaining option value to the directly executable one-minute action comparison.

No new market dates are opened. The already-opened Request-178 block is reused
only as a development diagnostic. A positive Request-179 result is not eligible
for promotion; it can only justify freezing a candidate for a later untouched
block.

## Frozen upstream policy

Unchanged:

- market-wide scan / Focus / Active / HOT hierarchy;
- Request-163 economic opportunity gate;
- Request-165 executability gate;
- BUY only when both upstream gates pass;
- Request-166 causal POSITION construction;
- Request-171 rich one-second aggregate features;
- Request-173 market-regime context;
- BASE execution-cost convention;
- 30-minute safety cap.

Request 179 does not change entry selection.

## Action target

For each causal POSITION state with an executable current exit and next-minute
exit reference:

    hold_advantage_1m_pct =
        BASE return if held exactly one more minute
        - BASE return if exited now

This target is already produced by the causal POSITION builder. Future prices
are labels only and never model inputs.

The candidate estimates the expected value of HOLD relative to EXIT directly.
The semantic action boundary is fixed:

- HOLD iff predicted expected one-minute advantage > 0;
- otherwise EXIT.

No threshold search is allowed.

## Candidate model

Use all currently available causal POSITION information:

- frozen minute/path POSITION features;
- Request-171 rich one-second aggregate features;
- Request-173 market-regime features.

Fit one HistGradientBoostingRegressor on the old fit partition only:

- squared error;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- minimum 75 samples per leaf;
- L2 regularization 2.0;
- random seed 20261079;
- fit-target winsorization at 0.5% / 99.5%.

Give each fit trading day equal total sample weight.

Use the chronological calibration partition only for one offset correction:
the equal-day-weighted mean of (actual advantage - raw prediction). Do not fit
a slope or tune a threshold.

## Recurrent execution diagnostic

On each Request-178 BUY episode:

1. require an observable minute-1 POSITION state;
2. at the reached state, score the candidate;
3. EXIT at the current executable reference when score <= 0;
4. HOLD exactly one minute when score > 0;
5. if the exact next causal state exists, score again;
6. if the next state is unavailable, exit at the already-defined next-minute
   executable reference and record a missing-state exit;
7. after minute 29, a HOLD forces the minute-30 safety-cap exit.

No state after an executed EXIT may influence the trajectory.

## Comparator

Reproduce the frozen Request-142 one-minute HOLD classifier on the same
fit/calibration partitions and the same episodes:

- HOLD iff calibrated P(one-minute HOLD wins) > 0.5;
- otherwise EXIT.

This isolates whether direct expected action value plus rich/regime information
improves the previously weak one-minute controller.

## Required diagnostics

Report:

- all-state expected-advantage Spearman;
- HOLD-selected rate and realized one-minute advantage;
- day-balanced selected advantage;
- per-day ranking and selected advantage;
- candidate and comparator completed trajectories;
- BASE arithmetic and day-balanced return;
- positive-trade rate, median, p05 and severe-loss rate;
- holding-time distribution and exit reasons;
- 10,000-sample trading-day bootstrap;
- matched candidate-minus-comparator day-balanced BASE difference and bootstrap;
- start-state coverage.

## Frozen development gate

Request 179 is a useful candidate only if all hold on the reused five-day
development block:

1. minute-1 start-state coverage >= 80%;
2. all-state advantage Spearman >= +0.05;
3. predicted-HOLD states realize positive mean one-minute advantage;
4. at least 3/5 days have both positive ranking and positive selected advantage;
5. candidate day-balanced BASE trade return > 0;
6. candidate day-balanced BASE return exceeds the old classifier;
7. matched candidate-minus-classifier day-balanced BASE difference > 0;
8. the matched trading-day bootstrap 95% lower bound > 0.

Even if all checks pass, promotion remains false because these dates were
already inspected in Request 178. The next action would be to freeze this exact
controller and preregister an untouched forward block.
