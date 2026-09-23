# Request 151 — fresh Active transport capacity confirmation

## Question

Requests 149 and 150 showed on the already-opened May 7-13 development block
that a five-addition Active transport ceiling may be binding. The eight-addition
sensitivity restored >=95% Focus -> Active retention while preserving nearly
full usable occupancy, but that result cannot itself be promoted because the
ceiling was compared after seeing the development block.

Request 151 therefore freezes the resource rule first and opens a new five-session
block only after the policy and gate are committed.

## Fresh evaluation block

Previously sealed sessions:

- 2026-05-21
- 2026-05-22
- 2026-05-26
- 2026-05-27
- 2026-05-28

The exchange holiday between May 22 and May 26 is skipped by the market calendar.
No later session is opened by this request.

## Policies

Both policies use exactly the same:

- market-wide Focus model;
- request-147 actionable admission score;
- request-149 three-minute transport-survival model;
- request-150 balanced stale-incumbent value:
  `active_priority * transport_survival_probability`;
- Active capacity 20;
- HOT capacity 10;
- one-second features;
- stage-2 model family and hyperparameters.

Only the transport ceiling differs.

### Primary

At each decision, replace only as many stale subscriptions as current desired
challengers require, capped at **8 additions**.

This does not force eight replacements. It is a demand-responsive ceiling.

### Comparator

Identical policy capped at the historical **5 additions**.

## Chronology

- Market-hazard / observability / transport-survival models remain fit only
  through 2026-01-16.
- Stage-2 fitting remains frozen to 2026-04-30 through 2026-05-06.
- The May 21-28 block is evaluation only.
- No threshold or hyperparameter is tuned after viewing this block.

## Fresh promotion gate

The primary can replace the five-addition ceiling only if all conditions pass:

1. corrected causal Focus capture conditional on observability >=95%;
2. Focus -> Active retention >=95%;
3. Focus -> Active retention is strictly greater than the five-addition comparator;
4. HOT conditional on Active >=90%;
5. one-second row coverage >=99%;
6. Active occupancy <=20 and HOT occupancy <=10;
7. actual post-initial additions average <=8 and maximum <=8;
8. selection/transport mismatch =0;
9. genuine sparse-slot waste <=2%;
10. genuine sparse-slot waste is no worse than the five-addition comparator;
11. mean scoreable Active occupancy >=19.60 / 20;
12. mean scoreable occupancy is no worse than comparator;
13. pooled HOT capture within 1m, 2m and 5m is non-lower than comparator;
14. HOT-within-1m is non-lower on at least four of five sessions.

If the gate fails, the eight-addition ceiling is not promoted.

## Interpretation

Passing would show that the previous five-addition ceiling was an artificial
transport bottleneck on a fresh period. It would not establish eight as a
permanent sacred constant. Eight would become the validated current resource
ceiling while later event-driven deployment can make observation capacity and
cadence explicitly resource-aware.

After a pass, downstream entry/opportunity evidence must be revalidated on
chronologically later sessions. Earlier May 14-20 evidence cannot be relabeled
as fresh after selecting the upstream transport policy using May 21-28.
