# Integrated learned-focus hierarchy v0.9 — pre-registration

## Status

Frozen after request 125 passed and before outcomes from 2026-03-04,
2026-03-05, 2026-03-06, 2026-03-09 or 2026-03-10 are inspected. Request-124
and request-125 evaluation dates are not reused for tuning. April 2026 and later
remain sealed.

## Question

Does the promoted learned market-wide focus-60 selector improve exact-prior-
minute HOT readiness after it is connected to the full top-20 -> targeted
one-second -> HOT-10 hierarchy?

## Frozen design

- Fit the market-wide focus hazard through 2026-01-16 with the request-125
  feature set and model hyperparameters.
- Rank all eligible rows each minute, admit at most 60, and send the top 20 by
  the same hazard probability to targeted one-second observation.
- Derive focus episode age only from consecutive learned-focus membership; it
  is causal and supplies the unchanged downstream feature contract.
- Fit the minute and one-second HOT rerankers on 2026-01-20 through 2026-02-09
  using the promoted selector's candidates and unchanged features/model.
- Build a same-date comparator by replaying the frozen pre-request-125 focus,
  top-20 and HOT hierarchy with its own chronologically fitted downstream model.
- Keep HOT capacity 10 and exact-tie incumbent preference unchanged.

## Metrics

Report pooled and by session:

- first +10% crossings;
- exact-prior-minute HOT capture for the frozen and integrated hierarchies;
- final-two-minute, final-five-minute and sustained-final-two-minute HOT
  readiness;
- integrated focus and top-20 exact-prior-minute capture;
- top-20 retention conditional on focus and HOT retention conditional on
  top-20;
- targeted one-second row coverage, HOT occupancy and turnover.

## Promotion gate

Promote the integrated attention hierarchy to HOT -> ENTRY/ABSTAIN research
only if all are true:

1. every evaluation session has at least 10 first +10% crossings;
2. pooled integrated exact-prior-minute HOT capture exceeds the frozen
   hierarchy on the same sessions;
3. pooled integrated exact-prior-minute HOT capture is at least 30%;
4. integrated capture is non-lower on at least four of five sessions;
5. integrated focus exact-prior-minute capture is at least 50%;
6. top-20 capture conditional on focus is at least 95%;
7. HOT capture conditional on top-20 is at least 90%;
8. one-second candidate-row coverage is at least 99%;
9. HOT occupancy never exceeds 10.

The 30% HOT floor is retained from request 124 rather than increased after the
request-125 result. If the gate fails, do not tune on these dates. Attribute the
loss to focus persistence, top-20 retention, second-data coverage or HOT
allocation and modify only the failed layer on a later untouched March block.

## Result — request 126

Request 126 completed successfully and passed every preregistered performance
condition. Across 1,034 first +10% crossings, exact-prior-minute HOT capture
rose from 281 (27.18%) in the frozen hierarchy to 529 (51.16%) in the integrated
hierarchy. The integrated policy improved on all five days; daily capture was
49.47%, 47.89%, 52.63%, 51.23% and 54.04%.

Integrated focus captured 574 runners (55.51%), top-20 retained 561 (97.74% of
focus captures), and HOT retained 529 (94.30% of top-20 captures). Final-two-
minute, final-five-minute and sustained-final-two-minute HOT readiness were
52.42%, 59.38% and 34.33%. One-second row coverage was 100% and HOT occupancy
never exceeded 10.

The performance promotion gate is true, but the operational audit exposes a
remaining deficiency. The integrated candidate set covered 11,327 ticker-days
versus 3,350 for the frozen hierarchy, required 8,741 uncached network requests,
and took about 67 minutes. HOT-set retention fell to 52.28%, with 4.79 of 10
slots replaced per minute. A persistence bridge is therefore required before
ENTRY/ABSTAIN integration: reduce top-20 observation demand and churn without
sacrificing the newly validated capture.
