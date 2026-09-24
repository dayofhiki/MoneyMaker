# Request 162 — fresh economic-opportunity revalidation on validated attention

## Purpose

Request 161 freshly validated the corrected attention handoff:

- learned Focus cap 180;
- three-minute Active model trained on fit-period Focus-180;
- Active features = BASELINE_FEATURES + market_hazard_probability;
- no market_rank;
- frozen Active threshold = **0.003093198572895277**.

Request 162 now returns to the previously blocked Request-148 question:
does first-HOT causal state still contain learnable economic-opportunity
information when HOT is fed by the validated attention architecture?

This is an economic-opportunity learnability test, not an executable trading
profit claim.

## Chronology

Economic-opportunity model development reuses only already-opened dates:

Fit:
- 2026-04-30
- 2026-05-01
- 2026-05-04
- 2026-05-05
- 2026-05-06

Calibration:
- 2026-05-07
- 2026-05-08
- 2026-05-11
- 2026-05-12
- 2026-05-13

Fresh evaluation opens exactly:
- 2026-06-08
- 2026-06-09
- 2026-06-10
- 2026-06-11
- 2026-06-12

June 1-5 remain validation evidence from Request 161 and are not used to tune
this experiment. No evaluation date may be replaced after outcomes are seen.

## Frozen attention and HOT path

Keep unchanged:

- market-hazard fitting through 2026-01-16;
- Focus cap 180;
- three-minute Active target and model;
- Active features = BASELINE_FEATURES + market_hazard_probability;
- Active threshold = 0.003093198572895277;
- dynamic Active membership, no fixed Active count;
- HOT budget = 10;
- existing one-second HOT reranker model family and fit window;
- existing first-HOT episode definition.

For the legacy HOT-reranker feature contract, the upstream stage-1 hazard
probability is the learned market_hazard_probability from the validated
attention path. State-age features are derived causally from consecutive
dynamic Active membership.

Every emitted dynamic-Active row is desired now; there is no retained-only
fixed-slot transport population.

## Frozen economic definition

Do not change Request-140B / Request-148 economics:

- one episode = first HOT for one ticker-day;
- entry reference = exact next minute-bar open after the HOT decision;
- candidate exits = observed positive minute opens through 30 minutes;
- unchanged BASE execution friction model;
- oracle_best_base_pct = best BASE round-trip return through 30 minutes;
- opportunity_positive = oracle_best_base_pct > 0;
- missing future economic outcomes remain unknown.

The oracle target is hindsight-only and is not an executable exit policy.

## Frozen ENTRY representation and models

Use the exact Request-140B ENTRY_FEATURES and the same classifier/regressor
families and hyperparameters.

Fit the classifier/regressor only on the already-opened fit days. Freeze the
regression calibration offset and the classifier top-quartile probability gate
on the already-opened calibration days.

No June outcome may affect a model, threshold, feature, date, or cost parameter.

## Promotion gate

Use the original Request-148 economic-opportunity gate unchanged:

1. at least 100 labeled first-HOT rows on every fresh evaluation day;
2. economic-label coverage >= 95% on every fresh day;
3. pooled classifier ROC AUC >= 0.55;
4. classifier ROC AUC > 0.50 on at least four of five fresh days;
5. pooled value Spearman >= 0.05;
6. value Spearman > 0 on at least four of five fresh days;
7. frozen selected set has positive mean oracle BASE value;
8. selected oracle mean strictly exceeds all-HOT oracle mean;
9. selected opportunity-positive rate exceeds all-HOT rate by >= 5 percentage points;
10. selected oracle mean is non-lower than daily all-HOT mean on at least four of five days.

Passing revalidates causal economic-opportunity ordering only. It does not prove
that an executable BUY/HOLD/EXIT policy makes money.

## Decision rule

If the gate passes, freeze this economic-opportunity representation and move to
causal executability/liquidity-aware BUY / WAIT / ABSTAIN and recurrent
HOLD / EXIT research.

If it fails, do not tune on June 8-12. Record which gate failed and diagnose
whether the weakness is population coverage, opportunity ordering, value
ranking, or selection economics before opening another fresh block.
