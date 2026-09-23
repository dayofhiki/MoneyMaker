# Request 143 — request-140B label-coverage diagnosis

## Status

Frozen after request 140B completed and before inspecting row-level causes of its
economic-label coverage shortfall.

Request 140B formally failed only because 2026-05-15 economic-label coverage
was 94.59%, below the frozen 95% floor by 0.41 percentage points. Predictive
ordering, daily AUC, value correlation and selection-effect conditions passed.

This diagnostic reuses the already-opened request-140B artifact only. It opens
no market date and does not retrain or retune any predictive model.

## Question

Why are some first-HOT episodes missing `oracle_best_base_pct`?

The deterministic request-139/140 labeler can produce a missing oracle label
only when:

1. the exact next-minute entry open is unavailable; or
2. the entry open exists but there is no later observed minute open through the
   30-minute research window.

The second case represents absence of an observable executable exit reference
under the current minute-bar data, not a prediction error.

## Frozen diagnostics

For the request-140B evaluation dates 2026-05-14, 05-15, 05-18, 05-19 and
05-20, report:

- total first-HOT rows, labeled rows and coverage;
- count missing because entry reference is absent;
- count with entry but no later exit reference;
- minutes-to-close distribution for labeled and missing rows;
- coverage by frozen minutes-to-close buckets: <=1, (1,2], (2,5], (5,10],
  (10,30], >30;
- current-price and active-seconds distributions for labeled and missing rows;
- same breakdown by evaluation day.

No threshold may be changed and request 140B may not be retroactively promoted.

## Decision rule

If the shortfall is dominated by non-executable/illiquid states, later policy
research must model executability explicitly and treat such states as abstain or
forced-risk cases rather than silently discarding them.

If missing labels occur broadly despite ample remaining session time and active
trading, investigate a timestamp/data-pipeline defect before opening May 21 or
later.

May 21 and later remain sealed.
