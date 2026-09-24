# Request 161 — fresh validation of the frozen hierarchical Active candidate

## Purpose

Requests 151-160 repeatedly used the same opened May development block. Request
160 produced the first clean Active handoff candidate:

- learned Focus cap: **180**
- temporal target: crossing within **3 minutes**
- temporal training population: fit-period learned Focus-180
- Active features: BASELINE_FEATURES + **market_hazard_probability**
- exclude market_rank
- frozen Active threshold: **0.003093198572895277**

Request 161 freezes that design and evaluates it once on chronologically later
sessions that have not appeared in the repository's prior research records.

## Fresh evaluation block

Open exactly:

- 2026-06-01
- 2026-06-02
- 2026-06-03
- 2026-06-04
- 2026-06-05

No date in this block may be replaced after outcomes are seen.

All supervised fitting remains restricted through **2026-01-16**.

## Frozen comparator

For context only, also evaluate the Request-158 Focus180 baseline:

- BASELINE_FEATURES only
- frozen threshold **0.003429286145316252**

No threshold is recalibrated on June.

## Primary metrics

Use the same exact-prior supported denominator used in development:

- Focus180 supported runner capture
- frozen candidate supported runner capture
- frozen baseline supported runner capture
- candidate-vs-baseline fresh capture delta
- Active count distribution
- 3-minute positive rate
- AP on Focus180
- rank 1-60 and rank 61-180 capture
- per-day Focus and candidate Active capture

Also report total runner crossings and the fraction not supported by the exact
prior-minute representation.

## Pre-registered gate

Require at least **100 supported exact-prior runner crossings** across the block.

**fresh_candidate_validated** if all hold:

1. Focus180 supported capture >= **99%**
2. frozen candidate supported capture >= **95%**
3. frozen candidate capture >= frozen baseline capture

**fresh_absolute_pass_feature_uplift_not_confirmed** if Focus and candidate meet
the 99% / 95% absolute gates but the candidate underperforms the baseline.

**fresh_focus_bottleneck** if Focus180 falls below 99%.

**fresh_active_candidate_failed** if Focus remains adequate but candidate falls
below 95%.

**fresh_validation_insufficient_support** if fewer than 100 supported crossings
are available.

No model, feature, threshold, cap, or date may be changed based on Request 161
results before the result is recorded.
