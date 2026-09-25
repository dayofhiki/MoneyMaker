# Request 210 — transition-aware recurrent HOLD/EXIT

## Status

Pre-registered after Request 208 showed that explicit causal state transitions
materially improved recurrent entry timing, while Request 209 showed that a
separate minute-1 admission hurdle did not solve profitability.

Request 179's direct recurrent HOLD/EXIT model failed to observe one-minute
continuation value: fresh advantage Spearman was -0.0191 and predicted-HOLD
states realized -0.0648% mean one-minute advantage.

No new dates are opened.

## Hypothesis

The HOLD/EXIT decision has the same temporal problem that entry timing had:
an isolated state does not explicitly tell the learner whether the move is
strengthening, stalling, pulling back, or recovering.

Keep every Request-179 causal input and add causal within-position transitions:

- one-minute change and acceleration of position return;
- one-minute change in drawdown from running peak;
- one-minute change in recovery from running trough;
- one-minute attention-score / rank changes;
- one-minute log-volume / log-transaction changes;
- one-minute active-second change;
- one-minute change in 5s and 10s momentum;
- one-minute change in close-vs-VWAP;
- one-minute change in short volume / transaction burst.

No future label, future price, or post-exit state becomes an input.

## Target

Unchanged from Request 179:

    hold_advantage_1m_pct =
        BASE return if held exactly one more minute
        - BASE return if exited now

Semantic action boundary remains fixed:

- HOLD iff predicted advantage > 0;
- otherwise EXIT.

## Comparator

Retrain the exact Request-179 direct expected-advantage model on the same
fit/calibration partitions. The only experimental change is the transition
representation.

## Frozen development gate

Transition HOLD/EXIT is useful only if all hold on the reused Request-178 block:

- minute-1 start coverage >= 80%;
- all-state hold-advantage Spearman >= +0.05;
- predicted-HOLD states realize positive mean one-minute advantage;
- at least 3/5 days have both positive ranking and positive selected advantage;
- transition controller day-balanced BASE return > Request-179 direct
  controller;
- matched transition-minus-Request179 day-balanced BASE difference > 0;
- matched day-cluster bootstrap 95% lower bound > 0;
- transition controller day-balanced BASE return > 0.

This request is development-only and promotion-ineligible.
