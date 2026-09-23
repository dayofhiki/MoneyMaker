# Rate-limited subscription tracking v1.4 — pre-registration

## Status

Fresh layer-validation experiment after request 133. Evaluation uses only
2026-04-16, 2026-04-17, 2026-04-20, 2026-04-21 and 2026-04-22. April 23 and
later remain sealed.

## Motivation

Request 133 showed that more responsive shortlist membership can recover the
predictive gate: bounded-turnover retained 573 of 599 focus captures (95.66%)
versus 566 / 599 (94.49%) for rank-40 hysteresis, and was non-lower on all five
days. But it averaged 8.01 new subscriptions per decision, failing the <=5
transport objective.

This indicates that model preference and transport state should be separated
more explicitly. The model should be free to name its current best 20
candidates immediately. The transport may temporarily retain old
subscriptions while moving toward that desired set under a bounded change
rate.

## Frozen design

- Fit the unchanged market-wide hazard model only through 2026-01-16.
- Keep focus capacity 60.
- Define the model's desired shortlist as the current score top-20 with no
  persistence.
- Maintain a separate active observation set of capacity 20.
- On the first decision of each session, initialize active membership to the
  desired top-20.
- Thereafter replace at most five active names per decision with the
  highest-ranked desired challengers not already active.
- Retained non-desired names are explicitly transport state, not model
  preference.
- Do not use a rank-40 expiry threshold in the active transport layer.
- Do not alter features, model fit, target label or focus budget.
- Do not fetch one-second data in this layer-validation experiment.

The limit of five is inherited unchanged from the previously frozen
operational objective and from request 133. It is not tuned on the new dates.

## Comparators and metrics

Compare:

1. request-131 rank-40 hysteresis;
2. current stateless desired top-20;
3. rate-limited active subscription set.

Report pooled and by day:

- focus exact-prior-minute capture;
- total exact-prior-minute capture for all three sets;
- retention of current focus captures by each observation set;
- mean and maximum additions per decision;
- mean set retention and ticker-day demand;
- mean active-versus-desired membership difference;
- maximum active occupancy.

## Promotion gate

Promote the transport layer only if all are true:

1. each day has at least 10 first +10% crossings;
2. pooled active capture exceeds the request-131 rank-40 comparator;
3. at least 95% of crossings captured by current focus are also active;
4. active capture is non-lower than rank-40 on at least four of five days;
5. active occupancy never exceeds 20;
6. mean additions per decision are <=5.

If the gate passes, the next untouched block may integrate this desired/active
separation with one-second data and HOT-10. If it fails, do not retune the
five-addition limit on these dates. Diagnose whether the remaining loss comes
from delayed challenger admission or from a capacity regime that requires a
different observation allocation strategy.

## Research integrity

This experiment is selected because request 133 isolated the tradeoff between
responsive membership and churn. April 23 and later remain untouched for full
integration or the next structural test.


## Result — request 135

Request 135 completed successfully on untouched 2026-04-16, 04-17,
04-20, 04-21 and 04-22 sessions after one shared lint-only retry that had not
opened the evaluation data.

Focus-60 captured 585 of 965 crossings (60.62%). The request-131 rank-40
selector retained 556 focus captures, while the rate-limited active set
retained 558. The new transport retained 95.04% of focus captures, exceeded the
rank-40 comparator in pooled exact-prior capture, was non-lower on four of five
sessions, and never exceeded 20 active subscriptions.

The formal gate remained false because mean additions were reported as 5.004
per decision versus the frozen <=5 floor. Inspection shows that this audit
includes the first decision of each of the five sessions, when the
preregistered design explicitly initializes all 20 subscriptions at once before
the five-addition rate limit applies. Total additions were 9,733 across 1,945
decisions. Subtracting only the five mandatory 20-name initial fills leaves
9,633 additions across 1,940 post-initial decisions, or 4.966 additions per
post-initial decision.

This is a measurement-definition mismatch, not grounds for retroactive
promotion. Request 135 remains formally failed. The policy parameters are not
changed. A fresh confirmation block must preregister the operational metric as
post-initial additions per decision, consistent with the original design text,
before opening new dates.

A secondary diagnostic remains: the active set differed from the instantaneous
desired top-20 by a mean symmetric-difference rate of 34.26%, yet retained
95.04% of focus-captured runners. That supports the architectural separation
between model preference and transport convergence rather than requiring the
transport to mirror every short-lived rank oscillation.
