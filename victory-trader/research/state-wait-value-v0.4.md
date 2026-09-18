# State Wait Value v0.4 — pre-registration

## Status

Pre-registered before implementation and before viewing any v0.4 result.

This experiment uses only the already-seen January-March 2026 development
panels. It must not load or inspect a fresh validation month.

## Motivation

State Entry Ranker v0.3 learned reproducible out-of-month within-episode timing
order, but its fixed top-25% score gate was too sparse to satisfy the minimum
trade-count rule. The exact rank-gate branch is retired and its 75th-percentile
threshold will not be tuned on January-March.

The next question is an optimal-stopping question:

> Conditional on the causal state available now, is waiting likely to offer a
> better after-cost entry than buying now?

## Fixed information set

Use exactly the same causal feature family as State Entry Ranker v0.3:

- point-in-time state features;
- contemporaneous leave-one-out runner breadth;
- the existing 36 causal ticker-local lag features;
- current and previous absolute price transforms.

No news, future bars as features, fresh month, severe-loss classifier, dynamic
stop branch, or new execution assumption is allowed.

## Buy action model

Retain the existing calibrated direct base-net EV regressors for 5-, 10-, and
15-minute actions.

The executable buy gate remains fixed at predicted best base-net EV >= 0.50%.

## Wait-value target

The wait horizon is fixed at 15 minutes.

For each eligible state in a ticker-day:

1. define current realized action value as the maximum realized base-net return
   across the 5-, 10-, and 15-minute buy actions at the current state;
2. examine only strictly later states in the same ticker-day whose timestamps
   are no more than 15 minutes after the current state;
3. define future realized action value as the maximum current-state action value
   among those future states;
4. define wait advantage as:
   future realized action value minus current realized action value;
5. states without both a valid current value and a valid strictly-future value
   inside the 15-minute window have no wait label and are excluded from wait
   model fitting.

Future returns are used only to construct the supervised training label. They
must never enter the feature frame.

## Wait model

Fit one histogram gradient-boosting regressor to the 15-minute wait-advantage
target.

- fit/calibration split: the existing chronological train-only split;
- state sampling: the existing three-minute training cadence;
- features: the fixed causal information set above;
- target winsorization: 0.5th and 99.5th percentiles on the fit sample;
- calibration: train-only affine calibration using the chronological
  calibration partition, with the same non-inverting calibration constraints as
  the direct EV model.

## Causal policies

### earliest_ev_cap1

Enter the first chronological state in a ticker-day whose best predicted direct
action EV is at least 0.50%. Choose the highest predicted-EV action horizon.
Permit at most one attempted entry per ticker-day.

### wait_value_cap1

At each chronological state:

1. require best predicted direct action EV >= 0.50%;
2. require calibrated predicted wait advantage <= 0.00 percentage points;
3. enter at the first state satisfying both conditions;
4. choose the highest predicted direct-EV action horizon;
5. permit at most one attempted entry per ticker-day.

The 0.00 wait threshold is semantic, not tuned: enter when the model no longer
expects waiting to improve the best available realized action value.

If the first accepted signal is unfillable or lacks the selected realized label,
the attempted entry is consumed. The policy must not fall through to a later
state using future label availability.

## Evaluation

Leave one month out, train on the other two, and repeat for January, February,
and March.

Report:

- trades, days, and ticker-day episodes;
- base, gross, and stress mean return;
- day-balanced base mean;
- p05, severe-loss rate, and worst day;
- entry minute and horizon mix;
- holdout wait-value diagnostics, including Spearman correlation and sign
  agreement between predicted and realized wait advantage;
- the same diagnostics restricted to states passing the 0.50% direct EV gate;
- common-episode timing and realized-return deltas against earliest_ev_cap1;
- pooled day-cluster bootstrap interval for wait_value_cap1.

## Success rule

Do not promote or consume a fresh month unless wait_value_cap1:

1. has at least 15 trades in every month;
2. has positive base-net mean in every month;
3. has positive day-balanced base-net mean in every month;
4. improves day-balanced base-net mean over earliest_ev_cap1 in every month;
5. does not worsen p05 or stress-net mean versus earliest_ev_cap1 in any month;
6. has positive holdout wait-advantage Spearman in every month in the
   EV-qualified state pool; and
7. has a pooled trading-day-cluster bootstrap 95% lower bound above zero.

If this branch fails, do not tune the wait threshold or the 15-minute window on
January-March. Retire the exact branch or change the information set/objective.
