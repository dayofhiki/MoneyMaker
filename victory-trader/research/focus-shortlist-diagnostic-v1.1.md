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


## Result — request 132

Request 132 completed successfully on the already-opened request-131 dates.

The key decomposition changes the interpretation of request 131. Across 1,089
first +10% crossings, only 557 had an eligible exact-prior-minute hazard row.
The learned focus-60 captured 538 of those 557 observable positives (96.59%),
which is the same 538 / 1,089 = 49.40% headline capture reported by request
131. Therefore the 50% focus gate sits very near the observability ceiling of
the completed-minute input on this block; the principal focus loss is missing
exact-prior-minute evidence, not poor rank ordering among rows that exist.

Among the 538 focus captures, the stateless current top-20 captured 497
(92.38%) and the request-131 rank-40 hysteresis selector captured 494 (91.82%).
Thus incumbency cost only three net captures. Fourteen positive rows were
current rank <=20 but blocked by incumbent priority, but hysteresis recovered
other positives outside the current top-20, offsetting most of that loss.

The pooled positive hazard-rank distribution was concentrated but not enough
for a fixed top-20 to satisfy the 95% conditional gate: 89.23% of exact-next-
minute positives were rank <=20, 94.43% rank <=40, 96.59% rank <=60. The
largest regime failure was 2026-04-08, where focus captured 133 runners but
both stateless and hysteresis top-20 captured only 107 (80.45% conditional).
On that day the positive rank p75 widened to 23.5 and p90 to 58.2.

Operationally, stateless top-20 averaged 8.56 additions per decision with
57.37% set retention, while rank-40 hysteresis averaged 4.63 additions and
77.02% retention. This confirms that simply removing persistence would violate
the frozen transport objective without solving the pooled 95% conditional
capture requirement.

Interpretation: request 131 has two distinct structural limits. First, a
completed-minute market-wide scanner cannot provide an exact prior row for a
large fraction of crossings, so higher-frequency/event-driven broad-market
observation is now an attention-recall research requirement rather than only a
deployment refinement. Second, fixed top-20 allocation can become too narrow
in broad opportunity regimes such as April 8. Request 133 was preregistered
before these results were read, so its bounded-turnover selector remains an
honest fresh-block test of the narrower incumbency hypothesis rather than a
response tuned to this diagnostic.
