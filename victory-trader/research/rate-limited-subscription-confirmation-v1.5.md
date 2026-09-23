# Rate-limited subscription confirmation v1.5 — pre-registration

## Status

Fresh confirmation of the unchanged request-135 transport policy.

Evaluation opens only 2026-04-23, 2026-04-24, 2026-04-27,
2026-04-28 and 2026-04-29. April 30 and later remain sealed.

## Why a confirmation is required

Request 135 kept the model's desired current top-20 separate from the active
subscription set and limited transport movement to at most five additions per
update after each session's initial fill.

On untouched 2026-04-16 through 2026-04-22 sessions the policy:

- retained 558 of 585 focus captures, or 95.04%;
- exceeded the rank-40 comparator's 556 captures;
- was non-lower on four of five days;
- never exceeded 20 active subscriptions.

The formal gate still failed because the audit averaged 5.004 additions per
decision against a <=5 floor. That statistic counted the mandatory first
decision of each session, which initializes all 20 subscriptions at once even
though the preregistered rate limit explicitly applies only after
initialization.

Request 135 remains failed. This confirmation does not reinterpret or promote
it.

## Frozen policy

No policy parameter changes are allowed.

- hazard fit remains frozen through 2026-01-16;
- focus capacity remains 60;
- desired membership remains instantaneous score top-20;
- active capacity remains 20;
- first decision of each session initializes active membership to desired
  top-20;
- every subsequent update may add at most five new names while moving active
  membership toward desired membership;
- no rank-40 expiry rule is added to the active layer;
- features, labels and model fitting remain unchanged;
- no one-second data are fetched in this layer confirmation.

## Corrected operational audit

The confirmation reports both old all-decision churn and the metric consistent
with the frozen policy text:

- mean additions per decision across all decisions;
- mean additions per **post-initial** decision;
- maximum additions per post-initial decision.

No first-session-fill cost is hidden; it is reported separately by implication
through the all-decision metric. It is simply not treated as a violation of a
rate limit that was never intended to apply to initialization.

## Promotion gate

Promote the desired/active transport separation only if all are true:

1. each evaluation day has at least 10 first +10% crossings;
2. pooled active exact-prior capture exceeds the rank-40 comparator;
3. at least 95% of crossings captured by focus are also active;
4. active capture is non-lower than rank-40 on at least four of five days;
5. active occupancy never exceeds 20;
6. mean post-initial additions per decision are <=5;
7. maximum post-initial additions per decision are <=5.

If the gate passes, the next untouched block may integrate the confirmed
transport with one-second observation and HOT-10. If it fails, do not tune the
five-addition limit on these dates; diagnose the delayed-admission loss before
choosing another structural transport policy.

## Research integrity

Request 136 changes no model or transport behavior from request 135. Only the
operational metric definition is aligned with the already-frozen initialization
exception, and that definition is fixed before the new dates are opened.


## Result — request 136

Request 136 passed every preregistered confirmation gate on untouched
2026-04-23, 04-24, 04-27, 04-28 and 04-29 sessions.

Across 600 first +10% crossings, focus-60 captured 349 (58.17%). The
rate-limited active set retained 341 of those focus captures (97.71%) versus
337 (96.56%) for the request-131 rank-40 selector. Active pooled exact-prior
capture was 56.83% versus 56.17% for rank-40 and was non-lower on all five
sessions.

The operational layer also passed cleanly:

- maximum active subscriptions: 20;
- mean post-initial additions per decision: 4.935;
- maximum post-initial additions per decision: 5;
- mean all-decision additions including mandatory session initialization:
  4.974;
- mean active/desired symmetric-difference rate: 25.68%.

The instantaneous desired top-20 itself churned much more heavily, averaging
7.624 post-initial additions per decision and reaching 14 in one decision.
This confirms the architectural result: the predictive attention layer should
be allowed to change its desired membership rapidly, while the observation
transport follows that target under an explicit rate limit.

Request 136 therefore promotes the desired/active separation and its
five-addition post-initial transport policy. The next fresh-block experiment
may integrate this confirmed observation set with the frozen one-second
reranker and HOT-10 runtime. ENTRY/ABSTAIN remains blocked until that complete
handoff passes.
