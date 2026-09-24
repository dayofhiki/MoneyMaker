# Request 158 — retrain temporal Active admission on Focus-180

## Question

Request 157 showed that Focus-60 is too narrow, but expanding Focus to 180
recovered 14 additional supported pre-runner crossings into Focus while only
four additional crossings reached the frozen Active policy.

One plausible cause is distribution mismatch: the three-minute Active model was
trained only on fit-period Focus-60 rows and therefore never learned the lower
market-hazard population newly exposed by Focus ranks 61-180.

Request 158 tests that hypothesis directly.

## Development-only scope

No new dates are opened. Evaluation remains restricted to the already-opened:

- 2026-05-21
- 2026-05-22
- 2026-05-26
- 2026-05-27
- 2026-05-28

All supervised fitting remains restricted through 2026-01-16.

Focus cap is frozen at **180** from Request 157.

## Models

Train two otherwise identical three-minute crossing models using the same
features and model hyperparameters:

1. baseline: fit-period learned Focus-60 rows
2. broad: fit-period learned Focus-180 rows

Both are evaluated on the same May Focus-180 population.

## Primary threshold comparison

The old baseline keeps Request 156's frozen threshold:

- **0.0040281217293971616**

Probability scale may shift after retraining, so using that same raw number as
the primary broad-model threshold would confound ranking quality with score
calibration.

Instead, before looking at May outcomes:

1. score fit-period Focus-180 with the baseline model;
2. measure the fraction / mean count admitted by the old threshold;
3. score the same fit-period Focus-180 rows with the broad model;
4. choose the broad-model threshold whose global fit-period score quantile
   reproduces that admission burden as closely as possible.

This fit-period burden-matched threshold is frozen before May evaluation.

The same raw numeric threshold is also reported as a secondary diagnostic only.

## Metrics

Primary comparison reports:

- supported exact-prior runner capture on Focus-180
- mean / median / p90 / p95 / max Active count
- three-minute positive rate among selected rows
- average precision on labelable Focus-180 rows
- separate capture for market-hazard ranks 1-60 and 61-180

The rank 61-180 bucket is the key mechanism test.

## Pre-registered success gate

A **focus180_training_candidate** requires all three:

- downstream supported capture >= 95%
- gain >= 0.50 percentage points versus the Focus-60-trained baseline
- mean May Active burden <= 110% of baseline

A gain >=0.50pp that misses another gate is
**focus180_training_partial_gain**.

Gain below 0.50pp is **focus180_training_no_material_gain** unless capture
regresses, which is **focus180_training_regression**.

This request is diagnostic only and does not open fresh validation dates.
