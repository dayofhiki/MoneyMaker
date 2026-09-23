# Bounded-turnover shortlist v1.2 — pre-registration

## Status

Frozen before request 132 diagnostic outcomes are inspected.

Request 131 exposed a structural selector defect independently of its outcome
decomposition: the rank-40 incumbent rule can retain all 20 incumbent slots
even when a fresh candidate becomes current hazard rank 1. This experiment
changes only that selector behavior.

The evaluation block is untouched 2026-04-09, 2026-04-10, 2026-04-13,
2026-04-14 and 2026-04-15. The already-opened April 1/2/6/7/8 dates are not
used to choose any new parameter. April 16 and later remain sealed.

## Question

Can the shortlist remain operationally stable while allowing newly strong
candidates to enter promptly?

## Frozen design

- Fit the unchanged market-wide next-minute hazard model only through
  2026-01-16.
- Keep the unchanged 60-name learned focus pool.
- Keep shortlist capacity 20 and the broad incumbent eligibility band at rank
  40.
- Compare the request-131 rank-40 incumbent selector against one structural
  change: bounded-turnover challenger admission.
- After the session's initial fill, any candidate currently inside the score
  top 20 may challenge a weaker retained incumbent.
- Admit at most five such replacements per decision. Five is inherited from
  the already-frozen transport resource objective of no more than five mean
  subscription additions per decision; it is not chosen from April outcomes.
- Deeply fallen or disappeared incumbents may vacate slots normally and do not
  consume the replacement budget.
- Do not change the hazard model, feature set, target label, focus budget,
  shortlist budget or fit dates.
- Do not fetch one-second data in this layer-validation experiment.

## Metrics

Report pooled and by session:

- first +10% crossings;
- focus-60 exact-prior-minute capture;
- request-131 rank-40 shortlist exact-prior-minute capture;
- bounded-turnover shortlist exact-prior-minute capture;
- both shortlist capture rates conditional on focus;
- non-lower capture days;
- maximum occupancy;
- mean and maximum additions per decision;
- mean set retention;
- ticker-day observation demand.

## Selector promotion gate

Promote only the shortlist layer if all are true:

1. every evaluation session has at least 10 first +10% crossings;
2. pooled bounded-turnover prior-minute capture exceeds the request-131
   rank-40 selector on the same sessions;
3. bounded-turnover capture conditional on focus is at least 95%;
4. bounded-turnover capture is non-lower on at least four of five sessions;
5. shortlist occupancy never exceeds 20;
6. mean additions per decision do not exceed 5.

Focus capture is reported but is deliberately not a gate in this experiment,
because request 133 changes only the shortlist layer. If the selector passes,
the next untouched block may integrate it with one-second reranking and HOT-10.
If it fails, do not tune the five-replacement budget on these dates; diagnose
whether the failure is insufficient responsiveness or unavoidable score
instability before selecting a different structural policy.

## Research integrity

Request 132 may continue running concurrently, but request 133's selector,
five-replacement resource limit, evaluation dates and gate are frozen before
request 132 results are read. Therefore request 132 cannot be used to tune
request 133.

April 16 and later remain sealed.


## Result — request 133

Request 133 completed successfully on untouched 2026-04-09, 04-10,
04-13, 04-14 and 04-15 sessions.

Focus-60 captured 599 of 987 crossings (60.69%). The request-131 rank-40
selector retained 566 of those focus captures (94.49%), while bounded-turnover
retained 573 (95.66%). Bounded-turnover therefore cleared the 95% predictive
retention floor, improved pooled exact-prior capture from 57.35% to 58.05%,
and was non-lower on all five sessions.

The promotion gate nevertheless failed on transport efficiency. Rank-40
hysteresis averaged 4.20 additions per decision with 79.20% set retention,
whereas bounded-turnover averaged 8.01 additions with only 60.09% retention.
Its five explicit challenger replacements did not cap total additions because
incumbents falling outside the broad eligibility band vacated slots and were
refilled outside that replacement count.

Interpretation: responsive challenger admission can recover the desired
attention retention, but combining model membership and transport membership
in one selector creates avoidable churn. The next layer test therefore
separates the current model-desired top-20 from the active subscription set
and rate-limits only the transport transition.
