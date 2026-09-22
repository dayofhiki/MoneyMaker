# Targeted one-second HOT reranker v0.5 — pre-registration

## Status

Frozen after request 120 failed only its preregistered per-day support floor and
before any v0.5 evaluation outcome is inspected.

This branch changes the causal information family by adding one-second path
information **only after** a frozen minute-hazard shortlist. It does not relax
the request-120 gate or declare v0.4 promoted. April 2026 and later remain
sealed.

## Architecture under test

The hierarchy is:

market-wide completed-minute scan
-> WATCH
-> frozen minute hazard shortlist
-> expensive one-second observation
-> HOT-10 ranking.

One-second aggregates are therefore no longer requested for every WATCH name.
They are reserved for names already judged urgent by a cheaper minute model.

## Frozen chronology

Three chronological blocks are fixed:

1. stage-1 shortlist model fit:
   2026-01-02 through 2026-01-16;
2. stage-2 reranker fit:
   2026-01-20 through 2026-02-02;
3. untouched evaluation:
   2026-02-03, 04, 05, 06 and 09.

The stage-1 model is fit only on block 1 and frozen before it generates
shortlists in blocks 2 and 3. This prevents same-sample shortlist leakage.

## Population and target

Use the exact recurrent WATCH population and target from v0.4:

- state is WATCH;
- ticker has not already crossed +10% from nominal prior close;
- an exact next regular-session minute exists;
- target = 1 only when the ticker's first +10% crossing occurs at the next
  minute decision.

## Stage-1 shortlist

Fit the exact v0.4 HistGradientBoostingClassifier and feature set on block 1.

At every timestamp in blocks 2 and 3:

- score all eligible WATCH rows;
- rank descending by minute hazard probability;
- retain at most the top 20.

This top-20 pool is frozen because it is twice the HOT budget of 10 and provides
room for a high-resolution reranker without returning to broad one-second
observation.

Report the fraction of all next-step positive WATCH events that survive into
the top-20 pool.

## One-second observation

For each ticker-day appearing in the frozen top-20 pool, fetch historical
one-second aggregates once and cache them.

At a candidate decision t, use only one-second bars whose full interval is
complete by t, exactly as in requests 115 and 116.

Reuse the frozen request-115 second-path feature definitions:

- active-second counts and last-activity age;
- 10s and 30s return shape/acceleration;
- realized second-close volatility;
- positive second-return fraction;
- max intra-minute drawdown/run-up;
- last-10s volume and transaction concentration;
- last-10s versus prior-10s volume and transaction burst.

Sparse seconds remain sparse. No forward filling and no future second bar are
allowed.

## Stage-2 comparison

On block 2 candidate rows, fit two fixed
HistGradientBoostingClassifiers with identical hyperparameters:

- minute reranker: v0.4 minute/state features plus frozen stage-1 hazard
  probability;
- second reranker: the same features plus the frozen one-second feature family.

Hyperparameters:

- learning_rate = 0.05
- max_iter = 120
- max_leaf_nodes = 7
- min_samples_leaf = 40
- l2_regularization = 1.0
- random_state = 17

No class weighting, threshold search, feature search or recalibration is
permitted after evaluation.

## Evaluation

On the untouched five-session block report:

- all-WATCH rows and positive events;
- top-20 candidate rows and positive events;
- top-20 positive-event coverage;
- one-second data coverage and request counts.

For minute and second rerankers report pooled and per-day:

- PR-AUC;
- ROC-AUC;
- Brier score;
- HOT-10 positive-event capture, measured against **all eligible WATCH positive
  events**, not only candidate positives;
- HOT-10 precision;
- mean rank of positive candidates within the top-20 pool.

## Promotion gate

Treat targeted one-second observation as useful for HOT allocation only if all
are true:

1. every evaluation session has at least 10 all-WATCH positive events;
2. frozen top-20 shortlist captures at least 95% of pooled all-WATCH positive
   events;
3. second reranker pooled PR-AUC exceeds minute reranker PR-AUC;
4. second reranker pooled HOT-10 all-WATCH capture is at least minute reranker
   capture;
5. second reranker HOT-10 capture is non-lower on at least four of five days;
6. second reranker pooled Brier is no worse than minute reranker Brier;
7. second reranker mean positive-candidate rank is lower (better).

Passing this gate promotes the hierarchical observation policy for a subsequent
chronological attention-runtime replay. It does not create an ENTRY or BUY
policy.

If the gate fails, do not tune top-20 size, second-feature windows or model
hyperparameters on these dates. The next branch should move high-resolution
information to the HOT -> ENTRY/ABSTAIN problem rather than repeatedly tuning
WATCH -> HOT.


## Result — request 122

Request 122 completed successfully after request 121 was retried unchanged
following minute-data integrity fixes.

Untouched evaluation covered 15,271 all-WATCH rows and 260 next-minute first
+10% crossings on 2026-02-03, 04, 05, 06 and 09.

The frozen stage-1 top-20 shortlist retained 258 of 260 positive events:

- pooled shortlist positive-event coverage: **99.23%**.

Pooled stage-2 comparison:

| Metric | Minute reranker | Minute + targeted 1s reranker |
| --- | ---: | ---: |
| PR-AUC | 0.268409 | **0.274595** |
| ROC-AUC | 0.944578 | **0.944969** |
| Brier | 0.0142469 | **0.0142426** |
| HOT-10 all-WATCH capture | 95.38% | **95.77%** |
| captured events | 248 / 260 | **249 / 260** |
| mean positive-candidate rank | 2.9690 | **2.8721** |

Second HOT-10 capture was non-lower on all five evaluation sessions.

The preregistered promotion gate **passes**.

Data audit:

- 2,624 candidate ticker-days requested;
- 2,624 returned non-empty second aggregates;
- candidate-row second-data coverage: 100%;
- 4,558,299 second aggregate rows;
- zero REST retries.

## Interpretation

This is the first preregistered evidence that targeted one-second observation
adds incremental value after a cheap minute-level attention model has already
narrowed the market.

The improvement is modest rather than transformative. The main value is
architectural: broad one-second observation is unnecessary, while selective
high-resolution observation of a frozen urgent shortlist improves ranking
without degrading any preregistered allocation criterion.

The promoted hierarchy is therefore:

market-wide completed-minute scan
-> WATCH
-> minute recurrent hazard shortlist
-> one-second observation of top-20
-> one-second reranked HOT-10.

The next experiment must place this hierarchy inside a **true chronological
attention runtime** on fresh development sessions. HOT membership must persist,
turn over and compete through time rather than being reconstructed as an
independent post-hoc top-10 at each timestamp. The runtime replay should compare
the frozen learned hierarchy against the existing baseline attention runtime,
measuring pre-crossing capture, lead time, HOT occupancy, churn and displacement
behavior.

Passing v0.5 does not create an ENTRY/BUY signal. One-second information is
promoted only as an attention-allocation input.
