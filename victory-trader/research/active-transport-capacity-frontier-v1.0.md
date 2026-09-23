# Request 152 — residual Active transport capacity frontier

## Why this experiment follows request 151

Request 151 improved the fresh May 21-28 transport state substantially when the
per-decision addition ceiling moved from five to eight:

- Focus -> Active retention: 92.53% -> 93.42%
- mean scoreable Active occupancy: 19.55 -> 19.95 / 20
- genuine sparse-slot waste: 2.06% -> 0.25%

But the 95% retention target still failed. Pooled HOT-within-1m also moved
slightly down, and only three of five sessions were non-lower than the
five-addition comparator.

That combination means the old five-name transport ceiling was real, but it does
not yet tell us whether the remaining miss is still caused by the eight-name
churn ceiling or by the admission/ranking policy itself.

## Diagnostic question

On the already-opened request-151 sessions, how much Focus -> Active retention is
recoverable purely by relaxing the transport ceiling while keeping every model,
score and retention value frozen?

## Data

This request opens no new date. It reuses only:

- 2026-05-21
- 2026-05-22
- 2026-05-26
- 2026-05-27
- 2026-05-28

Fit chronology remains unchanged:

- market hazard, observability and transport-survival fit only through 2026-01-16;
- no threshold or model is refit from May outcomes.

## Frozen policy

All variants use:

- the same market-wide Focus model;
- request-147 actionable admission score;
- request-149 three-minute transport-survival model;
- request-150 balanced stale-incumbent value:
  `active_priority * transport_survival_probability`;
- Active capacity 20.

Only the maximum post-initial additions per decision changes.

## Frontier

Evaluate the frozen policy at:

- 5
- 8
- 10
- 12
- 16
- 20

A ceiling of 20 is the minute-level full-refresh diagnostic: transport is no
longer allowed to block the current desired top-20 set.

For every point record pooled and per-day Focus -> Active retention, mean/min
scoreable occupancy, actual additions, selection/transport mismatch and
subscription-state waste.

## Preregistered interpretation

Target retention remains 95%.

- If cap=20 is still below 95%, classify the residual bottleneck as
  **selection_or_admission_bottleneck**. More transport capacity cannot solve it.
- If cap=20 reaches 95% and improves on cap=8 by at least 1 percentage point,
  classify it as **residual_transport_capacity_bottleneck**.
- If cap=20 reaches 95% but gains less than 1 point over cap=8, classify it as
  **mixed_or_near_plateau**.

Also report the smallest tested cap that reaches 95%, if one exists.

## Scope

This is a diagnostic-only request. It cannot promote a new production ceiling,
and it cannot be used as fresh validation because the May 21-28 block was
already opened by request 151. Its only job is to decide what the next model
intervention should target before any later session is opened.
