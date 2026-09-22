# Hierarchical HOT entry-readiness validation v0.7 — pre-registration

## Status

Frozen after request 123 completed and before any outcome from 2026-02-18,
19, 20, 23 or 24 is inspected. April 2026 and later remain sealed.

Request 123 is not retroactively promoted. Its formal gate failed because the
baseline captured zero runners, making its captured-only lead median undefined.
This branch avoids that ill-posed comparison on fresh sessions and tests the
more operational question required before HOT -> ENTRY integration.

## Question

Does the unchanged learned hierarchy keep future first +10% runners under HOT
observation immediately before crossing, rather than merely touching HOT at
some much earlier point in the session?

## Frozen policy

No policy or model parameter changes are permitted:

- focus capacity 60 with 0.60 / 0.40 entry/exit hysteresis;
- stage-1 fit: 2026-01-02 through 2026-01-16;
- stage-2 fit: 2026-01-20 through 2026-02-09;
- stage-1 top-20 shortlist;
- targeted completed one-second features;
- second-reranked HOT capacity 10;
- exact-tie incumbent preference;
- model features and HistGradientBoostingClassifier hyperparameters from v0.6.

The untouched evaluation sessions are 2026-02-18, 19, 20, 23 and 24.

## Metrics

For the unchanged baseline and learned hierarchy, report pooled and by session:

- first +10% crossings;
- HOT at the exact completed minute immediately before crossing;
- HOT at least once in the final two completed minutes;
- HOT at least once in the final five completed minutes;
- HOT throughout both final two completed minutes.

Decompose the learned policy at the immediately prior minute:

1. runner present in the causal focus pool;
2. runner retained in the frozen stage-1 top-20 shortlist;
3. runner allocated one of the HOT-10 slots.

Also retain second-data coverage, HOT occupancy and turnover audits.

## Promotion gate

Promote the frozen attention hierarchy to the subsequent HOT -> ENTRY/ABSTAIN
integration experiment only if all are true:

1. every evaluation session has at least 10 first +10% crossings;
2. learned pooled exact-prior-minute HOT capture exceeds baseline;
3. learned pooled exact-prior-minute HOT capture is at least 30%;
4. learned exact-prior-minute capture is non-lower than baseline on at least
   four of five sessions;
5. learned pooled final-two-minute capture exceeds baseline;
6. focus-pool exact-prior-minute capture is at least 50%;
7. shortlist exact-prior-minute capture conditional on focus is at least 95%;
8. one-second candidate-row coverage is at least 99%;
9. HOT occupancy never exceeds 10.

The 30% and 50% floors are operational readiness requirements, not thresholds
selected to fit the new sessions. Missed runners count as zero capture; no
captured-only statistic is required for the gate.

If the gate fails, do not tune on these dates. Attribute the loss to focus
admission, shortlist retention or HOT allocation and change only the failed
layer on a later untouched development block.

