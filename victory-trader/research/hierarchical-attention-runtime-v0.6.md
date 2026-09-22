# Hierarchical chronological attention runtime v0.6 — pre-registration

## Status

Frozen after request 122 passed the targeted one-second HOT reranker gate and
before any v0.6 evaluation outcome is inspected.

This experiment promotes the v0.5 hierarchy into a causal chronological
attention runtime. It does not create an ENTRY/BUY policy. April 2026 and later
remain sealed.

## Question

When the learned hierarchy actually controls the ten expensive HOT observation
slots through time, does it place future first +10% runners under HOT
observation earlier and/or more often than the original percentile-based
attention runtime?

## Architecture

The learned runtime is:

market-wide completed-minute scan
-> causal focus pool, capacity 60
-> recurrent minute-hazard top-20 shortlist
-> one-second observation only for that shortlist
-> one-second reranked HOT-10
-> remaining focus names stay WATCH.

The 60-name focus budget equals the original 50 WATCH + 10 HOT total attention
capacity. The cheap focus gate retains the original WATCH entry/exit hysteresis:

- enter focus at attention score >= 0.60;
- remain focused while score >= 0.40;
- capacity 60.

The cheap focus gate itself has no HOT state.

## Frozen chronology

Three model blocks:

1. stage-1 minute hazard fit:
   2026-01-02 through 2026-01-16;
2. stage-2 minute/second reranker fit:
   2026-01-20 through 2026-02-09;
3. untouched chronological runtime evaluation:
   2026-02-10, 11, 12, 13 and 17.

2026-02-16 is a US equity market holiday.

## Population and target

At each focus decision, include a ticker only when:

- it is in the causal focus pool;
- it has not already crossed +10% from nominal prior close;
- an exact next regular-session minute exists.

The stage-1/stage-2 target remains the v0.4/v0.5 next-step hazard:

target = 1 only when the ticker's first +10% prior-close crossing occurs at the
next minute decision.

## Features and models

Reuse the v0.5 model family and frozen feature definitions.

Stage 1:

- minute-state features;
- focus age;
- watch/focus-run age;
- fixed HistGradientBoostingClassifier hyperparameters from v0.5.

At every fit/evaluation timestamp retain at most the top 20 by stage-1 hazard.

Stage 2:

- minute reranker includes stage-1 hazard probability;
- second reranker additionally includes the frozen v0.5 one-second path
  features;
- fixed v0.5 HistGradientBoostingClassifier hyperparameters.

No feature, shortlist-size, HOT-size, threshold or hyperparameter search is
permitted after evaluation.

## Stateful HOT allocation

Process each evaluation session timestamp in ascending order.

At each timestamp:

1. update the cheap 60-name focus pool;
2. score all eligible focus names with stage 1;
3. retain stage-1 top 20;
4. attach only completed one-second information to those candidates;
5. rank by the frozen second reranker;
6. assign at most 10 names to HOT;
7. all other current focus names are WATCH.

The runtime stores the previous HOT set. Exact score ties prefer an incumbent
HOT name, then ticker alphabetically. This is a deterministic continuity rule,
not a tuned hysteresis bonus.

Crossed +10% names leave this attention-target population after the crossing;
the eventual full trader will hand them to ENTRY/POSITION logic.

## Comparator

Replay the existing baseline AttentionRuntime unchanged on the same evaluation
sessions:

- WATCH enter/exit 0.60 / 0.40;
- HOT enter/exit 0.85 / 0.70;
- max WATCH 50;
- max HOT 10.

## Primary evaluation

For baseline and learned runtimes report pooled and by session:

- first +10% runner crossings;
- crossings that were HOT strictly before the crossing decision;
- strict pre-crossing HOT capture rate;
- median and mean HOT lead minutes;
- fraction of all runners that became HOT at least two full minutes before
  crossing;
- maximum and mean HOT occupancy.

For the learned runtime also report:

- stage-1 top-20 next-step-positive coverage;
- second-data row coverage;
- HOT promotions;
- HOT demotions;
- mean minute-to-minute HOT-set retention;
- mean HOT slots replaced per decision.

## Promotion gate

Promote the hierarchical runtime as the attention policy for the next
HOT -> ENTRY/ABSTAIN research stage only if all are true:

1. every evaluation session has at least 10 first +10% runner crossings;
2. learned pooled HOT pre-crossing capture exceeds baseline;
3. learned HOT capture is at least baseline on at least four of five sessions;
4. learned pooled median HOT lead is at least baseline;
5. learned fraction captured at least two full minutes early exceeds baseline;
6. stage-1 top-20 next-step-positive coverage is at least 95%;
7. second-data row coverage is at least 99%;
8. HOT occupancy never exceeds 10.

If the gate fails, do not tune the frozen hierarchy on these dates. Analyze
whether the failure comes from focus gating, shortlist loss, high-resolution
reranking, or state turnover before choosing a new branch.


### Strict lead convention

For both runtimes, a crossing counts as attention-captured only if the ticker
was HOT at a decision timestamp strictly earlier than the first +10% crossing
timestamp. Becoming HOT on the already-completed crossing minute is not credited.
This matches the recurrent next-step hazard objective and prevents the crossing
bar itself from masquerading as advance attention.
