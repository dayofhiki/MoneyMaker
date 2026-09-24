# Request 159 — hierarchical Focus-context features for Active

## Question

Request 158 falsified the simple distribution-mismatch hypothesis: retraining
the three-minute Active model on Focus-180 did not improve capture of the
newly exposed market-hazard ranks 61-180. Both the old and broad-trained models
captured only 4 of 14 supported tail runners.

The next hypothesis is that the Active model is missing information created by
the upstream learned Focus model itself. A ticker enters Focus because of its
learned market-hazard probability and relative market-hazard rank, but the
three-minute Active model currently does not consume those values directly.

Request 159 tests whether passing that hierarchical context downstream improves
tail discrimination.

## Development-only scope

No new dates are opened. Evaluation remains:

- 2026-05-21
- 2026-05-22
- 2026-05-26
- 2026-05-27
- 2026-05-28

All fitting remains restricted through 2026-01-16.

Focus is frozen at Top-180.

## Controlled comparison

Both models are trained on the same fit-period Focus-180 population, use the
same 3-minute target and identical HistGradientBoosting hyperparameters.

Baseline features:
- the existing BASELINE_FEATURES

Hierarchical features:
- all baseline features
- market_hazard_probability
- market_rank

No other feature or target changes are permitted.

## Threshold matching

To isolate feature quality from observation burden, both models receive
thresholds derived only from fit-period Focus-180 scores.

The admitted fraction is frozen at the Request-158 fit-period baseline burden:

**0.1291012691012691**

Each model gets the probability cutoff corresponding to that same selected
fraction. May outcomes are never used to choose the cutoffs.

## Primary metrics

- supported exact-prior runner capture over the full market-supported denominator
- mean / median / p90 / p95 / max Active count
- 3-minute positive rate among selected Active rows
- average precision on labelable Focus-180 rows
- capture among market-hazard ranks 1-60
- capture among ranks 61-180

The 61-180 bucket is the mechanism test.

Request 158 reference:
- broad Focus-180 model: 647/684 overall
- tail ranks 61-180: 4/14

## Pre-registered interpretation

**hierarchical_focus_context_candidate**
requires all of:
- overall supported capture >= 95%
- at least 7/14 tail runners captured
- mean Active burden <= 110% of baseline

**hierarchical_focus_context_partial_tail_gain**
if tail capture improves by at least 2 runners but the full gate is not met.

**hierarchical_focus_context_no_mechanism_gain**
if tail capture improves by fewer than 2 runners without regression.

**hierarchical_focus_context_regression**
if overall capture or tail capture falls.

This request is diagnostic only. No later unseen session is opened.
