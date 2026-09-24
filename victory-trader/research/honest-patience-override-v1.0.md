# Request 180 — honest patience override on recurrent EXIT baseline

## Status

Pre-registered after Request 179 failed and before running Request 180.

Request 179 established that directly predicting the next one-minute HOLD-minus-
EXIT return from the full causal feature set does not transfer: the fresh
advantage Spearman was negative and broad recurrent HOLDing worsened BASE
returns. At the same time, Request 178 established that medium-horizon
remaining-opportunity ranking does transfer better than its absolute zero
boundary.

Request 180 therefore preserves the strongest executable short-horizon
behavior as the default and gives the medium-horizon signal only one narrow
role: rescue a baseline EXIT for exactly one minute when an independently
trained patience model expects that intervention to add BASE value.

No new market dates are opened. June 23/24/25/26/29 remain development-only
because they were already opened by Request 178.

## Frozen upstream inputs

Unchanged:

- Request-165 BUY population;
- Request-166 causal POSITION rows and BASE execution references;
- Request-171 rich one-second features;
- Request-173 market-regime context;
- Request-176/177 consensus hurdle-EV construction;
- Request-142 calibrated one-minute HOLD classifier;
- 30-minute safety cap.

## Honest calibration split

Sort the chronological calibration trading days and split once:

- Calibration-A = first half;
- Calibration-B = second half;
- require at least four trading days in each half.

Using fit + Calibration-A only, train:

1. the Request-142 short HOLD classifier;
2. the row-weighted hurdle-EV heads;
3. the equal-day-weighted hurdle-EV heads.

Score Calibration-B with those frozen models.

## Honest patience target

For each Calibration-B episode, walk backward through its causal POSITION
states and compute the realized value of the frozen short-HOLD teacher.

At a state where the teacher says HOLD, teacher value is the actual downstream
value obtained by following the teacher at the next exact state. If the next
state is missing, use the already-defined next-minute executable BASE reference.

At a state where the teacher says EXIT, teacher value is EXIT-now BASE return.

For teacher-EXIT states only:

    patience_override_advantage_pct =
        actual value of forcing HOLD exactly one minute
        then following the frozen teacher
        - EXIT-now BASE return

This target is constructed only on Calibration-B, which the teacher did not use
for fitting or calibration.

## Override model

Inputs are exactly:

- calibrated short HOLD probability;
- conservative consensus hurdle EV = min(row-weighted EV, equal-day-weighted EV).

No raw market/path features are allowed into the override model.

Fit one HistGradientBoostingRegressor on valid Calibration-B teacher-EXIT rows:

- squared error;
- learning rate 0.05;
- 120 iterations;
- 7 leaves;
- minimum 100 samples per leaf;
- L2 2.0;
- seed 20261080;
- target winsorization at 0.5% / 99.5%.

Require at least 500 valid honest override targets.

No output calibration and no threshold search.

## Final development policy

Retrain the frozen upstream short-HOLD and consensus heads on the complete old
fit + calibration history, without using Request-178 outcomes.

On each reached Request-178 POSITION state:

1. if short HOLD probability > 0.5, HOLD;
2. otherwise predict patience override advantage;
3. if predicted override advantage > 0, HOLD for exactly one minute;
4. otherwise EXIT now;
5. after HOLD, recompute at the next actually reached state;
6. force exit at 30 minutes.

Comparator: unchanged Request-142 recurrent short-HOLD policy.

## Required diagnostics

Report:

- honest Calibration-B target count, mean, positive rate and model fit range;
- fresh short-HOLD and override-HOLD decision rates;
- candidate/comparator completion coverage;
- BASE mean, day-balanced mean, median, p05, positive rate and severe-loss rate;
- holding-time distribution and exit reasons;
- 10,000-sample trading-day bootstrap;
- matched candidate-minus-comparator difference and bootstrap.

## Interpretation gates

Controller-improvement gate requires:

1. start-state coverage >=80%;
2. completion coverage >=90%;
3. matched candidate-minus-comparator day-balanced BASE difference >0;
4. matched difference bootstrap 95% lower bound >0;
5. candidate severe-loss rate no worse than comparator.

Economic gate additionally requires:

6. candidate day-balanced BASE return >0;
7. candidate BASE day-bootstrap 95% lower bound >0.

This request is never promotion-eligible because the evaluation dates are
already opened.

If controller-improvement passes but economic gate fails, the next branch moves
upstream to entry abstention/value against realized recurrent-policy returns
instead of continuing to tune HOLD/EXIT.

If controller-improvement fails, reject the patience-rescue hypothesis and do
not tune the zero boundaries on Request-178 dates.
