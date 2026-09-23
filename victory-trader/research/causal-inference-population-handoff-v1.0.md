# Request 146 — causal inference population handoff revalidation

## Status

Frozen after the second recurrent-trader implementation audit and before
inspecting any corrected handoff result.

No new market date is opened. Request 146 reuses 2026-05-07, 05-08, 05-11,
05-12 and 05-13, the already-opened request-138 block. May 21 and later remain
sealed.

## Defect being corrected

The historical market-hazard helper constructed supervised rows by requiring an
exact next-minute aggregate. That requirement is legitimate for identifying the
next-minute training label, but historical inference reused the same filtered
row population.

At decision time a trader cannot know whether a ticker will print an aggregate
in the following minute. Therefore exact-next-minute availability is future
information and must never determine whether the current ticker enters market
ranking, Focus, active observation or HOT.

This can create selection bias even when runner prior-minute rows themselves are
labelable, because future-silent competing tickers are removed from the
cross-sectional ranking.

## Corrected population rule

For every current pre-runner market row:

- causal features are constructed from information available through the
  current completed bar only;
- the row remains eligible for inference regardless of future bar existence;
- `target_next_cross` is nullable;
- an exact next-minute bar is used only to decide whether that row may enter a
  supervised fit/evaluation label.

Market-hazard and stage-2 models fit only rows with identifiable labels.
Inference and attention allocation score all causal rows.

## Frozen architecture

- market hazard fit cutoff remains 2026-01-16;
- stage-2 fit remains 2026-01-20 through 2026-02-09;
- Focus budget 60;
- active subscription budget 20;
- max post-initial additions 5;
- HOT budget 10;
- same one-second features and stage-2 model hyperparameters;
- explicit subscription-state transport;
- active age persists through temporary unscoreable rows.

No threshold or model hyperparameter is tuned on May 7-13.

## Required audit output

Report:

- total causal inference rows versus exact-next-minute labelable rows;
- fraction of causal current rows that the historical inference population
  would have removed;
- raw runner crossings;
- Focus exact-prior capture;
- Focus capture conditional on a causal exact-prior feature row;
- active exact-prior capture and retention conditional on Focus;
- HOT within 1m / 2m / 5m and sustained metrics;
- HOT conditional on active observation;
- one-second row coverage;
- explicit subscription capacity/churn;
- per-day versions of the above.

## Frozen revalidation gate

The corrected handoff passes only if:

1. Focus capture conditional on causal observable prior rows >= 95%;
2. active retention conditional on Focus >= 95%;
3. HOT conditional on active exact-prior capture >= 90%;
4. pooled HOT-within-1m raw crossing rate >= 30%;
5. one-second data row coverage >= 99%;
6. max concurrent active subscriptions <= 20;
7. max HOT occupancy <= 10;
8. mean post-initial subscription additions per decision <= 5;
9. max post-initial additions <= 5;
10. selection/transport mismatch rows = 0.

This gate does not claim profitability. It only decides whether the previously
promoted attention/HOT handoff survives removal of the future-label-availability
selection leak.

If it fails, the attention/HOT layer is reopened and downstream opportunity and
position evidence is treated as population-conditional until rebuilt.

If it passes, the corrected causal population becomes the only allowed
population for subsequent opportunity/ENTRY/recurrent-policy research.

May 21 and later remain sealed.
