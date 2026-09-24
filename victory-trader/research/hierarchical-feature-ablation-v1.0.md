# Request 160 — hierarchical feature ablation

## Why

Request 159 produced a mixed result. Adding both upstream Focus-context
features raised overall supported runner capture from 647/684 (94.59%) to
652/684 (95.32%) while reducing mean Active burden, but rank-61-180 tail
capture regressed from 4/14 to 3/14.

That means hierarchical context is useful overall, but the specific claim that
it fixes tail discrimination is not supported. Request 160 isolates the two new
features to determine whether probability and rank play different roles.

## Scope

No new dates are opened. Use only the already-opened May block:

- 2026-05-21
- 2026-05-22
- 2026-05-26
- 2026-05-27
- 2026-05-28

All fitting remains through 2026-01-16.

Focus remains Top-180 and the 3-minute target is unchanged.

## Feature variants

Train four otherwise identical models on the same fit-period Focus-180 rows:

1. baseline
2. baseline + market_hazard_probability
3. baseline + market_rank
4. baseline + market_hazard_probability + market_rank

## Fair thresholding

Each model receives a threshold derived only from fit-period Focus-180 scores
to select the same frozen fraction:

**0.1291012691012691**

This isolates feature quality from simply observing more names.

## Metrics

For every variant report:

- supported exact-prior runner capture
- mean / median / p90 / p95 / max Active count
- 3-minute positive rate
- average precision
- rank 1-60 capture
- rank 61-180 capture

Baseline tail reference is **4/14**.

## Interpretation

A **hierarchical_ablation_candidate** must:
- reach at least 95% overall supported capture,
- capture at least 4/14 tail runners, so it does not regress the tail,
- remain within 110% of baseline mean Active burden.

Among clean candidates choose the highest overall capture, then higher tail
capture, then lower burden.

If a model reaches 95% only while tail capture falls below 4/14, diagnose
**overall_gain_tail_tradeoff**.

Otherwise diagnose **hierarchical_features_not_decisive**.

This is development-only; no fresh dates are opened.
