# Request 157 — Focus cap frontier

## Question

Request 156 showed that a rigid Active-20 budget is a bottleneck. Before opening
fresh dates, test whether the upstream learned Focus-60 ceiling is also
arbitrary.

This request optimizes only the **fixed Focus count**. It does not yet replace
Focus Top-K with an absolute threshold.

## Frozen downstream policy

To isolate the Focus ceiling, keep Request 156's winning Active rule unchanged:

- signal: Focus-specific P(runner crossing within 3 minutes)
- threshold: **0.0040281217293971616**
- source: Request 156 fit-q65 threshold
- temporal model training population: fit-period learned Focus-60 only

The threshold is not re-fit or re-selected using May outcomes.

## Development data

No new dates are opened. Reuse only:

- 2026-05-21
- 2026-05-22
- 2026-05-26
- 2026-05-27
- 2026-05-28

All supervised fitting remains restricted through 2026-01-16.

## Focus caps

Compare market-hazard Top-K Focus at:

- 20
- 30
- 45
- 60
- 75
- 90
- 120
- 180
- unbounded

The market-hazard model and ranking score are identical across policies.

## Comparable capture denominator

Do not compare only Active retention conditional on Focus because changing K
changes that denominator.

For every runner crossing with an exact prior-minute market row, report:

1. Focus supported capture
2. final dynamic-Active supported capture
3. Active capture conditional on Focus

Also report mean/median/p90/p95/min/max Focus and Active counts and the
three-minute positive rate among selected Active rows.

## Pre-registered knee rule

Let the best tested Focus supported capture define the Focus frontier and the
best tested downstream Active supported capture define the end-to-end frontier.

A finite cap is considered near-max only if it is simultaneously:

- within **0.25 percentage points** of the best Focus supported capture, and
- within **0.50 percentage points** of the best downstream Active supported capture.

Choose the **smallest finite cap** satisfying both. This deliberately prefers
the smallest observation budget once capture is effectively saturated.

Diagnosis:

- `focus_cap_reduction_supported`: knee < 60
- `focus60_near_knee`: knee = 60
- `focus_cap_increase_supported`: knee > 60
- `focus_cap_ceiling_unresolved`: no finite tested cap reaches both near-max
  criteria

This remains development-only. The winning architecture is frozen before any
chronologically later unopened validation session is opened.
