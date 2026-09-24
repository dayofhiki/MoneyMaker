# Request 165 — causal entry executability gate

## Goal

Move the validated attention/economic-opportunity stack one step closer to an
executable recurrent trader. Request 164 showed that 97.14% of missing fresh
economic labels were caused by the absence of an exact next-minute entry bar,
not by session close. Request 165 therefore models entry observability directly.

This request is an entry bridge. It does not freeze a holding period and does
not claim profitability.

## Frozen upstream system

Unchanged:

- Request-160/161 market-wide Focus-180 + dynamic Active architecture;
- Active feature set including market_hazard_probability and excluding
  market_rank;
- Active threshold 0.003093198572895277;
- existing one-second HOT reranker;
- HOT budget 10;
- Request-163 economic-opportunity classifier family and top-quartile
  calibration rule;
- Request-163 development history only.

## Executability model

Target:

- 1 when an exact next executable minute entry reference exists;
- 0 otherwise.

Inputs:

- exactly the existing causal ENTRY_FEATURES;
- no future bar, return, economic label, or execution outcome enters features.

Model:

- HistGradientBoostingClassifier;
- learning_rate 0.05;
- max_iter 160;
- max_leaf_nodes 15;
- min_samples_leaf 60;
- l2_regularization 2.0;
- random_state 20261065.

Fit dates:

- 2026-04-30, 05-01, 05-04, 05-05, 05-06, 05-07, 05-08.

Calibration dates:

- 2026-05-11, 05-12, 05-13, 05-14, 05-15, 05-18, 05-19, 05-20.

The execution threshold is selected only from calibration executability
outcomes. Choose the lowest threshold that retains the largest population with:

- observed entry-reference rate >= 90%;
- at least 200 calibration rows.

Economic outcomes do not participate in this threshold selection.

## Runtime action semantics

At first HOT:

- BUY = frozen economic-opportunity gate passes AND executability gate passes;
- WAIT = economic-opportunity gate passes but executability gate fails;
- ABSTAIN = economic-opportunity gate fails.

WAIT here is an entry-readiness state, not yet the final recurrent policy.
Request 166 will make WAIT/HOLD/EXIT recurrent if this bridge passes.

## Fresh sessions

Previously unopened:

- 2026-06-15
- 2026-06-16
- 2026-06-17
- 2026-06-18
- 2026-06-22

No threshold/model/feature/date changes are allowed after these sessions open.

## Promotion gate

Pass the entry bridge only if all are true:

1. at least 25 BUY decisions on every fresh day;
2. BUY exact-next-minute entry observability >= 90% pooled;
3. BUY exact-next-minute entry observability >= 85% on every fresh day;
4. fresh executability ROC AUC >= 0.55;
5. BUY oracle-positive rate is no more than 2 percentage points below the
   frozen economic-only selector;
6. BUY oracle mean is no more than 0.25 percentage points below the frozen
   economic-only selector.

Mechanical BASE-net returns at 1,2,5,10,15,30 minutes are reported only as
diagnostics. No horizon may be selected from this fresh block and called
validated.

If this passes, proceed to a recurrent HOT -> BUY/WAIT/ABSTAIN and
POSITION -> HOLD/EXIT development branch on already-opened history, followed by
a later unopened fresh block.
