# Request 149 — retention-aware Active resource allocation

## Purpose

Request 147 improved Focus -> Active retention and HOT capture but failed to
reduce wasted high-resolution capacity. Its one-minute observability factor was
too myopic for a subscription that may occupy a slot for several decisions.

Request 149 separates two decisions:

1. **Admission**: which Focus-60 names should enter the desired Active-20?
2. **Retention / eviction**: among stale incumbents, which subscriptions should
   be released first when only five new additions are allowed?

No new market dates are opened. May 21+ remain sealed.

## Frozen primary architecture

- Focus ranking: unchanged market-hazard model.
- Admission priority: unchanged request-147
  `market_hazard_probability * next_minute_observability_probability`.
- New desired Active candidates: current Focus-60 only.
- Active capacity: 20.
- Post-initial additions: maximum 5 per decision.
- HOT capacity: 10.
- Stage-2 one-second reranker and hyperparameters: unchanged.

## Retention model

A separate three-minute transport-survival classifier is fit only on dates
through 2026-01-16 using the frozen causal BASELINE_FEATURES.

The target is positive when, from the current pre-runner state, the ticker:

- graduates into runner state before a sparse wall-clock gap, or
- remains continuously observable through the full three-minute transport
  horizon.

A sparse missing bar before either outcome is negative. Rows too near session
close are unlabeled unless runner graduation happens first.

The three-minute horizon is a transport-value label, not a final trading or
holding horizon. Runtime is still recurrent and reevaluates every decision.

When challengers exceed the five-addition budget, stale incumbents with the
lowest predicted transport-survival value are evicted first. A currently
missing incumbent has no retention score and is released before scoreable
incumbents.

## Corrected slot audit

Historical `scoreable_now=False` mixed two different states:

- runner graduation: not wasted attention;
- current raw market bar absent: genuine sparse slot waste.

Request 149 reports these reasons separately. Promotion is gated on the genuine
sparse-waste rate, not the mixed historical unscoreable rate.

## Frozen evaluation

Primary evaluation: already-opened 2026-05-07, 05-08, 05-11, 05-12, 05-13.

Request 149 passes only if:

1. the standard causal handoff gate passes;
2. Focus -> Active retention >= 95%;
3. genuine sparse-waste rate is lower than the request-147 policy re-evaluated
   on the same rows;
4. HOT-within-1m >= 50.2569%;
5. HOT-within-2m >= 59.1984%;
6. HOT-within-5m >= 65.5704%;
7. second-data coverage >= 99%;
8. max Active <= 20, max HOT <= 10;
9. post-initial additions average <= 5 and max <= 5;
10. selection/transport mismatch = 0.

## Sensitivity only

The same retention policy is also replayed with max additions 6, 8 and 20.
Those variants are diagnostic only and cannot be promoted. They answer whether
the five-addition transport ceiling itself is materially binding.

## Next step

If primary request 149 passes, unblock request 148 and revalidate first-HOT
economic opportunity on the corrected Active/HOT population.

If it fails, use the reason audit and churn sensitivity to decide whether the
remaining bottleneck is retention-value estimation, admission ranking, or the
physical transport ceiling. Do not open May 21+.
