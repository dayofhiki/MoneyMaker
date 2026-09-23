# Focus / shortlist miss diagnostic v1.1

## Status

Diagnostic-only bridge after request 131 failed its frozen promotion gate.

The only evaluation sessions inspected here are the already-opened request-131
sessions: 2026-04-01, 2026-04-02, 2026-04-06, 2026-04-07 and 2026-04-08.
No later April session is opened by this diagnostic.

## Question

Where did request 131 lose runner capture?

The failed conditions were:

- learned focus-60 exact-prior-minute capture: 49.40% versus a 50% floor;
- shortlist capture conditional on focus: 91.82% versus a 95% floor.

The downstream HOT layer, one-second coverage and observation transport gates
passed. This diagnostic therefore isolates the two upstream losses without
changing the fitted model, feature set, focus budget, shortlist budget or
hysteresis rule.

## Frozen diagnostic

1. Fit the unchanged market-wide next-minute hazard model only through
   2026-01-16.
2. Score the already-opened request-131 April sessions with the unchanged
   request-131 feature set and model.
3. Record the exact rank of every next-minute first +10% crossing.
4. Compare only two already-existing shortlist policies:
   - stateless current top-20 by the same hazard score;
   - the request-131 event-driven rank-40 incumbent hysteresis selector.
5. Measure whether a runner that is currently in hazard rank 1-20 is excluded
   by the incumbent-priority hysteresis selector.
6. Report churn for the existing stateless and request-131 selectors.

This is not a selector search. No new rank buffer, capacity, score margin,
feature, label or threshold is evaluated on these dates.

## Outputs

Report pooled and by day:

- focus-60 exact-prior-minute capture;
- stateless top-20 and request-131 hysteresis top-20 capture;
- conditional top-20 capture given focus;
- positive rank distribution at 20 / 40 / 60 / 80 / 100;
- median, p75 and p90 positive rank;
- count of next-minute runners that were current hazard rank <=20 but were
  blocked by request-131 hysteresis;
- shortlist retention, additions per decision and ticker-day demand.

## Interpretation rule

The diagnostic may identify which structural layer failed, but its April
numbers must not be used to tune a new numeric threshold.

The next changed policy must be preregistered before opening any later April
session. The first candidate untouched block remains 2026-04-09,
2026-04-10, 2026-04-13, 2026-04-14 and 2026-04-15.

If incumbent priority blocks materially relevant current top-20 runners, the
next experiment should change the selector structurally so fresh high-ranked
challengers have guaranteed access while retaining a persistence region.

If focus misses are mostly deep beyond rank 60, the next experiment should
change the causal focus objective/features rather than simply increase the
focus budget.
