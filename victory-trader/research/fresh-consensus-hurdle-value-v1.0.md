# Request 178 — fresh consensus hurdle-EV validation

## Purpose

Requests 176 and 177 independently produced strong but not fully stable
post-entry expected-value signals on June 8-12. Their bootstrap lower bounds
remained slightly below zero, and the weak day differed between estimators.

Do not choose whichever June estimator looked better.

Request 178 freezes a conservative cross-estimator consensus **before opening a
new block**:

    consensus_EV = min(
        row_weighted_hurdle_EV,
        equal_day_weighted_hurdle_EV
    )

A state is consensus-positive only when both EV estimators are positive.

This request validates observability only. It does not yet execute a recurrent
HOLD/EXIT trajectory.

## New fresh block

Previously unopened:
- 2026-06-23
- 2026-06-24
- 2026-06-25
- 2026-06-26
- 2026-06-29

No date, feature, threshold, weighting rule, model capacity or target may be
changed after these sessions open.

## Frozen upstream pipeline

Unchanged:
- validated market scan / Focus / Active / HOT stack;
- Request-163 economic-opportunity model;
- Request-165 executability gate;
- BUY iff both economic and executability gates pass;
- Request-166 causal POSITION construction;
- Request-171 rich one-second aggregate features;
- Request-173 market-regime features.

## Frozen consensus heads

Estimator A:
- exact Request-176 row-weighted three-head hurdle EV.

Estimator B:
- exact Request-177 equal-trading-day weighted three-head hurdle EV.

Both are trained only on the old Apr/May development partitions.

## Fresh target

Keep the same diagnostic target:

    excess_remaining_option_value_pct

The target is still oracle-style remaining opportunity inside the inherited
30-minute safety cap and is not an executable stopping rule.

## Frozen promotion gate

Request 178 passes only if all hold on the new five-day block:

1. BUY-anchor to causal POSITION path coverage >=85%;
2. market-context coverage >=95%;
3. Spearman(consensus EV, realized excess remaining value) >= +0.08;
4. consensus-positive states are >=10% of evaluable states;
5. selected realized excess mean >= +0.20%;
6. selected day-balanced realized excess mean >0;
7. trading-day cluster-bootstrap 95% lower bound >0;
8. both daily ranking and selected realized excess are positive on >=4/5 days.

Do not relax the gate after inspection.

A pass promotes only the **post-entry value head**. The next step must still
construct an honest one-minute-at-a-time executable stopping controller before
claiming profitable HOLD/EXIT behavior.
