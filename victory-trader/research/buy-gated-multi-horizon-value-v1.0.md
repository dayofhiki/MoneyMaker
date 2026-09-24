# Request 169 — direct multi-horizon continuation value

## Purpose

Request 168 showed that adding the same path-transition feature family to the
oracle-style remaining-value regressor materially worsened June 8-12 ordering.

Request 169 changes the POSITION target instead of expanding features.

At every causal POSITION state, estimate the incremental BASE value of exiting
exactly 2, 5, or 10 minutes later relative to EXIT now. These are lookahead
prediction targets, not fixed holding commitments. Any eventual policy must
re-evaluate after every reached minute.

## Data

No new market date is opened.

Reuse Request-166 BUY-gated position rows:
- fit: Apr 30 through May 8;
- calibration: May 11 through May 20;
- evaluation diagnostic: June 8-12.

## State

Keep the corrected Request-166 causal MODEL_FEATURES exactly unchanged.
Do not add Request-167 path transitions.

## Targets

For each horizon h in {2,5,10} minutes, when the exact same episode has a
POSITION state at t+h:

    hold_advantage_h =
        BASE return from EXIT at exact t+h
        - BASE return from EXIT now

Missing exact future states remain missing. No timestamp compression or future
fallback enters the target.

Fit one HGB regressor per horizon with:
- squared error;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- minimum leaf size 75;
- L2 2.0;
- fit-only 0.5%/99.5% winsorization;
- one chronological calibration mean-bias offset.

## Predeclared multi-horizon diagnostic

At evaluation time, choose the horizon with the largest predicted advantage.
This is a score diagnostic only.

For that predicted horizon, compare its realized exact-horizon advantage.

The bridge passes only if:

1. chosen-horizon realized target coverage >=70%;
2. Spearman(predicted chosen advantage, realized chosen advantage) >=0.05;
3. predicted-positive chosen states are >=10% of evaluable states;
4. their day-balanced realized chosen advantage is positive;
5. trading-day cluster bootstrap 95% lower bound is positive;
6. daily Spearman and selected realized advantage are both positive on at least
   4/5 evaluation sessions.

No horizon may be selected post hoc from the result.

If this passes, the next experiment may use the frozen three-model vector as a
one-minute-at-a-time recurrent patience controller on already-opened data. It
must not commit to the diagnostic horizon.
