# Request 172 — uncertainty-aware rich-second patience value

## Purpose

Request 171 modestly improved remaining-value ranking and, importantly, moved
the selected-set trading-day bootstrap lower bound from negative to positive.
However it failed the preregistered material-improvement gate. Its point
estimate remained over-confident on June 11.

Request 172 does not add features or change the target. It asks whether a
fit/calibration-only conservative lower bound can define a robust patience set.

No new market dates are opened.

## Data

Reuse Request-171 enriched phase artifacts:
- fit: Apr 30 through May 8;
- calibration: May 11 through May 20;
- evaluation diagnostic: June 8-12.

## Target and state

Unchanged:
- Request-166 excess remaining-option target;
- original causal POSITION state plus the fixed Request-171 rich-second
  aggregate features.

## Frozen uncertainty method

Fit one HistGradientBoostingRegressor on the fit partition with:
- quantile loss;
- quantile = 0.20;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- min leaf 75;
- L2 2.0;
- deterministic seed;
- fit-only 0.5%/99.5% target winsorization.

On chronological calibration only:
1. predict the raw 20th-quantile value;
2. compute residual = actual excess value - raw quantile prediction;
3. take the 20th percentile of calibration residuals using the lower empirical
   quantile;
4. add that fixed correction to all later lower-bound predictions.

Runtime semantic candidate:

    PATIENCE-CANDIDATE iff calibrated lower bound > 0

No June outcome changes the quantile, correction rule, or zero boundary.

## Frozen development bridge

Pass only if all hold on June 8-12:

1. empirical lower-bound coverage >=70%;
2. selected states are >=5% of evaluable states;
3. selected realized excess mean >0;
4. selected positive-target rate >=55%;
5. selected day-balanced excess >0;
6. selected trading-day bootstrap 95% lower bound >0;
7. selected realized excess mean >0 on all 5 evaluation days.

This is development-only. If it passes, the next experiment may compose a
one-minute-at-a-time executable trajectory: HOLD exactly one more minute when
the conservative lower bound is >0, then recompute at the next reached state;
otherwise EXIT. It must be tested first on already-opened data before any new
fresh block.
