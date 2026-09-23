# Active transport integrated HOT v1.6 — pre-registration

## Status

Fresh full-handoff validation after request 136 promoted the desired/active
observation transport.

Evaluation opens only 2026-04-30, 2026-05-01, 2026-05-04,
2026-05-05 and 2026-05-06. May 7 and later remain sealed.

## Question

Does the promoted rate-limited active observation set preserve or improve the
complete Focus -> 1-second -> HOT-10 handoff when compared with the prior
rank-40 integrated hierarchy?

## Frozen architecture

The following upstream components remain unchanged:

- market-wide hazard fit ends 2026-01-16;
- Focus capacity remains 60;
- desired high-resolution membership is current hazard top-20;
- active high-resolution capacity remains 20;
- active membership moves toward desired membership with at most five new
  subscriptions per post-initial update;
- stage-2 fit remains 2026-01-20 through 2026-02-09;
- HOT capacity remains 10;
- one-second feature definitions and model hyperparameters remain unchanged.

For the active path, a temporarily retained subscription may leave the current
Focus-60 before transport converges. Its current minute feature row therefore
comes from the broad scored market rather than from the Focus-only table.
Focus/watch age features are defined as the consecutive active-observation
episode age for this path. The stage-2 models are retrained only on the same
historical fit period under that deployment-consistent candidate definition.

The comparator reproduces the prior learned Focus-60 -> rank-40 persistent
shortlist -> one-second reranker -> HOT-10 hierarchy on the same fresh dates.

## Observability-aware evaluation

Request 134 showed that many raw first +10% crossings have no causal
pre-threshold observation window. This experiment therefore reports both:

1. raw first-crossing recall against all nominal +10% crossings;
2. Focus capture conditional on a valid exact-prior market-hazard row existing.

The second quantity is the controllable scanner-quality gate. This does not
remove jump events from the audit; raw recall remains fully reported.

## Metrics

Report pooled and by day:

- total first +10% crossings;
- exact-prior observable crossing count and rate;
- raw Focus prior-minute capture;
- Focus capture conditional on observability;
- rank-40 and active prior-minute capture;
- rank-40 and active retention of Focus-captured crossings;
- rank-40 and active HOT capture within 1, 2 and 5 minutes;
- two-minute sustained HOT capture;
- HOT retention conditional on active observation;
- one-second row coverage;
- HOT occupancy and churn;
- active transport occupancy, post-initial additions and mismatch audit.

## Promotion gate

Promote the integrated active handoff only if all are true:

1. every evaluation session has at least 10 first +10% crossings;
2. pooled active HOT-within-1m capture exceeds the rank-40 integrated
   comparator on the same sessions;
3. active HOT-within-1m raw capture is at least 30%;
4. active HOT-within-1m capture is non-lower on at least four of five sessions;
5. active HOT-within-2m capture is at least the rank-40 comparator;
6. Focus captures at least 95% of causally observable exact-prior crossings;
7. active observation retains at least 95% of Focus-captured crossings;
8. HOT retains at least 90% of active prior-minute captures;
9. one-second data coverage is at least 99%;
10. HOT occupancy never exceeds 10;
11. active subscription occupancy never exceeds 20;
12. active transport has zero active-row mismatch;
13. mean post-initial subscription additions per decision are <=5;
14. maximum post-initial additions per decision are <=5.

If the gate passes, the attention/observation handoff is considered validated
enough to unblock the next policy-integration step: ENTRY/ABSTAIN on HOT
candidates. If it fails, no entry model is opened; diagnose whether the loss
originates in active transport lag, second-level reranking or HOT allocation.

## Research integrity

No date in the evaluation block is used to choose the architecture, feature
set, transport limit or gate. May 7 and later remain sealed.


## Result — request 137

Request 137 completed successfully on untouched 2026-04-30, 05-01,
05-04, 05-05 and 05-06 sessions. The formal promotion gate failed, but the
failure is isolated to the transport audit rather than predictive attention or
HOT allocation.

Across 1,087 first +10% crossings:

- 648 had a causal exact-prior market-hazard row;
- Focus-60 captured 634 of those 648 observable crossings (97.84%);
- active observation retained 605 of the 634 Focus captures (95.43%);
- HOT-10 captured 574 runners at the exact prior minute (52.81%);
- the rank-40 integrated comparator captured 566 (52.07%);
- active HOT-within-1m was non-lower on all five sessions;
- active HOT-within-2m was 53.82% versus 53.27% for rank-40;
- HOT retained 94.88% of active prior-minute captures;
- one-second row coverage was 100%;
- HOT occupancy never exceeded 10;
- active subscription occupancy never exceeded 20;
- selection/transport mismatch rows were zero.

Thus every predictive and observation-quality gate passed.

The formal failure came from the post-initial transport audit:

- mean measured additions per decision: 5.0005 versus the <=5 gate;
- maximum measured additions in one decision: 7 versus the <=5 gate.

Inspection of the implementation shows that this is not evidence that the
RateLimitedSubscriptionSelector changed more than five subscriptions. That
selector hard-caps new active names at five and retains active names even when
they are absent from the current candidate universe. The integrated audit,
however, reconstructs subscription membership from feature rows that exist in
the current scored-market table. If an already-subscribed ticker temporarily
has no eligible market-hazard row, it disappears from the reconstructed
'actual' set; when its row reappears later, the audit counts it as a new
subscription even though the selector never removed it. This can create
apparent additions above five.

Therefore request 137 remains formally failed and is not retroactively
promoted. The next experiment should change no ranking, focus, transport limit,
second-level model or HOT rule. It should make transport state explicit across
temporarily missing feature rows, separately audit subscription membership from
scoreable candidate membership, and then rerun the same integrated handoff on a
new untouched block.

The current evidence does not justify returning to attention-model tuning:
observable Focus recall, active retention, second-data coverage and HOT
allocation all passed their frozen gates.
