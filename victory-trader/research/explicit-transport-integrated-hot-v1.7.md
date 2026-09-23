# Explicit subscription-state integrated HOT v1.7 — pre-registration

## Status

Fresh confirmation after request 137 isolated its only formal failure to
transport-state accounting.

Evaluation opens only 2026-05-07, 2026-05-08, 2026-05-11,
2026-05-12 and 2026-05-13. May 14 and later remain sealed.

## What request 137 established

Request 137 passed every predictive and observation-quality gate:

- Focus captured 97.84% of causally observable exact-prior crossings;
- active observation retained 95.43% of Focus captures;
- active HOT-within-1m capture was 52.81% versus 52.07% for rank-40;
- active HOT-within-1m was non-lower on all five days;
- active HOT-within-2m was 53.82% versus 53.27%;
- HOT retained 94.88% of active prior-minute captures;
- one-second row coverage was 100%;
- HOT occupancy stayed <=10 and active occupancy <=20.

Its formal gate failed because the row-reconstructed churn audit reported
5.0005 mean post-initial additions and a maximum of seven. The selector itself
hard-caps additions at five. The discrepancy occurs when an already-subscribed
ticker temporarily has no scoreable market-hazard row: row reconstruction
treats the missing row as a subscription removal and its later reappearance as
a new addition.

Request 137 remains failed and is not retroactively promoted.

## Frozen policy

No predictive or allocation behavior changes are allowed.

- market-wide hazard fit remains through 2026-01-16;
- Focus capacity remains 60;
- desired high-resolution membership remains instantaneous hazard top-20;
- active subscription capacity remains 20;
- the first session decision initializes active membership;
- later updates may add at most five new subscriptions;
- stage-2 fit remains 2026-01-20 through 2026-02-09;
- one-second features, reranker model settings and HOT capacity 10 are unchanged;
- rank-40 integrated hierarchy remains the comparator.

## Only implementation change

Subscription membership is now recorded directly from the transport selector.

A ticker can therefore be:

1. actively subscribed and scoreable now;
2. actively subscribed but temporarily missing a scoreable minute row;
3. not actively subscribed.

Temporary feature-row absence does not imply unsubscribe/resubscribe.

The HOT reranker still scores only rows for which its frozen feature set exists.
No stale feature imputation or new model feature is introduced.

## Required diagnostics

Report both audits:

- **explicit subscription audit:** membership directly from selector state;
- **legacy row-reconstructed audit:** membership inferred from scoreable rows.

Also report:

- temporarily unscoreable subscribed rows and rate;
- mean/minimum scoreable active occupancy;
- true mean and maximum post-initial subscription additions;
- all request-137 Focus, active, second-data and HOT metrics.

## Promotion gate

Use the unchanged request-137 performance gates, but transport churn is measured
from explicit selector state:

1. every evaluation session has >=10 first +10% crossings;
2. active HOT-within-1m exceeds rank-40 integrated HOT;
3. active HOT-within-1m raw capture >=30%;
4. active HOT-within-1m is non-lower on >=4/5 sessions;
5. active HOT-within-2m is at least rank-40;
6. Focus captures >=95% of causally observable exact-prior crossings;
7. active observation retains >=95% of Focus captures;
8. HOT retains >=90% of active prior-minute captures;
9. one-second row coverage >=99%;
10. HOT occupancy <=10;
11. true active subscriptions <=20;
12. explicit selector/transport mismatch rows = 0;
13. mean post-initial true subscription additions <=5;
14. maximum post-initial true subscription additions <=5.

The legacy row-reconstructed churn numbers are diagnostics only.

## Decision rule

If request 138 passes, the attention/observation handoff is promoted and the
next active research milestone becomes HOT -> ENTRY/ABSTAIN integration.

If request 138 fails any predictive gate, do not open ENTRY/ABSTAIN. Attribute
the loss to the failed layer. If only explicit transport state fails, repair
transport state without changing the predictive hierarchy.

May 14 and later remain sealed.
