# Request 150 — balanced Active retention value

## Purpose

Request 149 nearly eliminated genuinely wasted Active slots, but it did so by
evicting stale subscriptions using only their predicted ability to remain
observable for three minutes. That improved usable occupancy while reducing
Focus -> Active runner retention.

Request 150 changes only the stale-incumbent eviction value. Admission remains
unchanged.

No new market dates are opened. May 21+ remain sealed.

## Frozen architecture

- Focus ranking: unchanged.
- New Active admission priority: unchanged request-147 actionable priority.
- New desired Active names: current Focus-60 only.
- Active capacity: 20.
- Primary post-initial additions: maximum 5 per decision.
- HOT capacity: 10.
- One-second HOT reranker: unchanged.
- Three-minute transport-survival model: unchanged request-149 model and
  training dates.

## Single primary change

Request 149 stale eviction value:

    transport_survival_probability

Request 150 stale eviction value:

    active_priority * transport_survival_probability

This requires an incumbent to be both economically interesting as an emerging
runner candidate and likely to remain observable. A missing incumbent still
has no retention value and is released first.

The three-minute survival label remains a resource-management label, not a
fixed trading or holding horizon.

## Evaluation block

Reuse only the already-opened sessions:

- 2026-05-07
- 2026-05-08
- 2026-05-11
- 2026-05-12
- 2026-05-13

No threshold, model hyperparameter, capacity, or churn budget is tuned on these
sessions.

## Frozen promotion gate

Request 150 passes only if all of the following hold:

1. the standard corrected causal handoff gate passes;
2. Focus -> Active retention >= 95%;
3. genuine sparse-slot waste <= request-149 primary value
   0.018487179487179487;
4. mean usable Active occupancy >= request-149 primary value
   19.603589743589744;
5. HOT-within-1m >= request-147 value 0.5025693730729702;
6. HOT-within-2m >= request-147 value 0.591983556012333;
7. HOT-within-5m >= request-147 value 0.6557040082219938;
8. one-second coverage >= 99%;
9. max Active <= 20 and max HOT <= 10;
10. mean post-initial additions <= 5 and maximum <= 5;
11. selection/transport mismatch = 0.

## Sensitivity only

The same balanced retention rule may be replayed with addition limits 6, 8 and
20. Those results are diagnostic only and cannot be promoted.

The purpose is to distinguish a bad eviction rule from a genuinely binding
five-addition resource ceiling.

## Consequence

If the primary request passes, the corrected Focus -> Active -> HOT handoff is
unblocked and request 148 may revalidate first-HOT economic opportunity on that
population.

If the primary fails while the eight-addition diagnostic passes strongly,
treat the five-addition ceiling as an architectural resource constraint rather
than continuing to tune ranking scores against the same development block.
Freeze a new resource-budget comparison before opening a fresh block.
