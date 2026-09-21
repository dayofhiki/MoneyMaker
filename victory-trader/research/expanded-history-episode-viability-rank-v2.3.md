# Expanded-history episode viability + timing rank v2.3 — pre-registration

## Status

Pre-registered after exact v2.2 was retired and before implementing or viewing
any v2.3 model output.

Use only September 2025 through March 2026 development panels. April 2026 and
later remain sealed.

This branch tests one structural intervention: separate ticker-day tradeability
from within-episode entry timing. It removes the failed absolute predicted-EV
+0.50% gate while preserving a direct after-cost economic target.

## Motivation fixed before results

Expanded-history v2.2 produced a complete February result before January/March
operationally failed. The exact branch was already promotion-ineligible in
February: only two rank-policy BUY attempts, one evaluated, with -2.4806% base
return. The absolute EV gate produced only five earliest-EV evaluated trades,
with -7.9763% mean base return.

At the same time, February's within-episode rank signal remained positive and
ordered by horizon:

- 5m global Spearman 0.0762; episode median 0.1111;
- 10m global 0.1086; episode median 0.1996;
- 15m global 0.1347; episode median 0.2415.

Across prior experiments the 15-minute rank signal was also the strongest and
most persistent. This branch therefore fixes the action horizon at 15 minutes
and asks two distinct causal questions:

1. does the first eligible state indicate that a fixed 15-minute trade in this
   ticker-day is more likely than not to be profitable after BASE costs?
2. if yes, when inside that episode does the 15-minute relative-rank model say
   the entry quality has reached the calibration-period upper quartile?

This is adaptive development research, not confirmatory evidence. The choices
above are frozen before v2.3 outcomes and must not be revised on Jan-Mar results.

## Frozen data and information set

Development panels:

- 2025-09
- 2025-10
- 2025-11
- 2025-12
- 2026-01
- 2026-02
- 2026-03

For every model explicitly use the existing multi-source causal feature frame:

- point-in-time price/volume/breadth features;
- 36 causal ticker-local sequence features;
- current and previous absolute-price transforms;
- exact lagged FINRA short-volume features;
- exact strictly-prior 8-K features;
- exact publication-safe FINRA short-interest features.

The implementation must call the multi-source feature builder, not merely load
the external columns.

No NBBO feature is allowed because configured historical quote REST and Flat
File paths returned HTTP 403. No news text, new external source, feature subset
search, hand interaction search, cost change, fresh validation month, brokerage
action, or threshold tuning is allowed.

## Strict past-only folds

Evaluate only:

- January 2026: train/calibrate on Sep-Dec 2025;
- February 2026: train/calibrate on Sep 2025-Jan 2026;
- March 2026: train/calibrate on Sep 2025-Feb 2026.

Within each fold, sort all strictly-prior scoreable training days and use the
chronological first 80% as fit days and final 20% as calibration days.

The workflow must persist a provenance table proving
max(training/calibration day) < min(evaluation day).

## Stage 1 — episode viability

An episode is one ticker-day.

For every ticker-day, define the anchor as its first chronological state passing
the existing causal state-eligibility rule. Future-label availability must not
choose the anchor.

Training label at that anchor:

- 1 if the realized 15-minute BASE-net return is strictly > 0%;
- 0 if the realized 15-minute BASE-net return is <= 0%;
- missing if the fixed 15-minute BASE-net label is unavailable.

Missing labels are excluded from model fitting only and must remain explicit in
evaluation.

Fit one HistGradientBoostingClassifier on fit-day anchor rows:

- loss="log_loss";
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=15;
- min_samples_leaf=75;
- l2_regularization=2.0;
- random_state=20261001;
- no class weights.

Probability calibration is fixed as chronological Platt scaling:

1. score calibration-day anchor rows with the fitted classifier;
2. clip raw probabilities to [1e-6, 1-1e-6];
3. regress the binary calibration labels on raw log-odds with one unpenalized
   logistic regression;
4. if calibration has fewer than 100 labeled anchors or lacks either class,
   fail the fold rather than invent a fallback.

Executable viability rule:

- arm the ticker-day iff calibrated P(BASE-net 15m > 0 | anchor state) >= 0.50.

The 0.50 threshold is semantic and must not be tuned.

## Stage 2 — within-episode timing rank

Use only the fixed 15-minute action.

For fit days, compute each state's percentile rank of realized 15-minute
BASE-net return within its own ticker-day using the full eligible trajectory
before applying the already-existing three-minute training cadence.

Fit one HistGradientBoostingRegressor with the frozen v0.3 rank settings:

- loss="squared_error";
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=31;
- min_samples_leaf=300;
- l2_regularization=2.0;
- random_state=20261035.

Calibration-day states remain full resolution.

Rank gate:

- score all eligible calibration states;
- require at least 15 finite scores;
- fixed gate = calibration predicted-rank 75th percentile.

No realized calibration return is used to choose the gate.

## Executable policies

### earliest_eligible_15m_cap1

Attempt a 15-minute BUY at the first eligible state of every ticker-day.

### earliest_viable_15m_cap1

At the first eligible state, compute calibrated episode viability.

- if probability < 0.50: SKIP;
- if probability >= 0.50: attempt a 15-minute BUY immediately.

At most one attempt per ticker-day.

### viability_rank_15m_cap1 — primary

At the first eligible state, compute calibrated episode viability.

- if probability < 0.50: SKIP the ticker-day permanently;
- if probability >= 0.50: arm the episode.

For an armed episode, process eligible states chronologically and attempt the
first 15-minute BUY whose predicted episode-rank score is >= the frozen
calibration 75th-percentile gate.

If no state reaches the gate, SKIP.

The first BUY signal consumes the ticker-day attempt even if entry/reference or
future evaluation labels are missing. Do not fall through based on future-label
availability.

## Execution and accounting

Keep existing light/base/stress cost scenarios unchanged.

Persist every BUY attempt with mutually exclusive evaluation reason.

Replay all three policies through the frozen position ledger:

- independent synthetic account per evaluation month/scenario;
- unchanged $10,000 accounting unit and $1,000 fractional order unit;
- no leverage or daily cash reset;
- delayed exits, unresolved capital, same-ticker blocks and missing references
  handled exactly by the existing ledger.

These are accounting conventions, not capital recommendations or observed
brokerage fills.

## Required diagnostics

For every evaluation month report:

- date provenance;
- anchor count and labeled-anchor coverage;
- anchor positive rate;
- raw and calibrated viability probability summaries;
- viability ROC AUC, balanced accuracy, log loss and Brier score;
- predicted viable-episode rate;
- 15m global rank Spearman;
- 15m median within-episode Spearman and positive-correlation rate;
- rank gate;
- attempts/evaluated/unevaluable and mutually exclusive missing reasons;
- gross/base/stress means;
- day-balanced base mean, median, p05, severe-loss rate and worst day;
- common-episode return delta for primary versus both comparators;
- external-data coverage;
- pooled trading-day-cluster bootstrap for the primary policy;
- position-ledger light/base/stress marked return and complete-accounting status.

## Development promotion rule

Do not access April 2026 or any later month unless ALL conditions hold for
viability_rank_15m_cap1:

1. at least 15 evaluated trades in January, February and March;
2. arithmetic BASE-net mean > 0 in every month;
3. day-balanced BASE-net mean > 0 in every month;
4. day-balanced BASE-net mean strictly exceeds both
   earliest_eligible_15m_cap1 and earliest_viable_15m_cap1 in every month;
5. BASE p05 and STRESS mean are not worse than earliest_eligible_15m_cap1 in
   any month;
6. viability ROC AUC > 0.50 and 15m rank Spearman > 0 in every month;
7. pooled trading-day-cluster bootstrap 95% lower bound > 0;
8. short-volume latest-prior coverage >=90%, publication-safe short-interest
   latest coverage >=90%, and 8-K query completion =100% in every month;
9. BASE-scenario marked account return > 0 and complete_accounting=true in
   every month.

A sparse profitable-looking month cannot pass. A light-scenario profit cannot
substitute for BASE. Zero trades cannot pass. Missing references/outcomes cannot
be assigned zero P&L.

## Failure rule

If v2.3 fails, do not tune the 15-minute horizon, 0.50 viability gate, 75th
rank gate, classifier/ranker capacity, class weights, cost scenarios or feature
subset on Jan-Mar.

Use the frozen diagnostics to decide between:

- candidate-universe redesign if episode viability is not learnable;
- execution-protocol/data improvement if gross alpha is positive but BASE is
  systematically erased by friction;
- cross-sectional capital allocation if viable episodes are learnable and
  cost-positive but concurrent candidate selection is the bottleneck.

Do not add HOLD/SELL freedom until a cost-positive entry process replicates.


## Result — request 65

Workflow run 35553113572 completed successfully. All three strict past-only
monthly jobs completed and the frozen position ledger replay produced final
artifact `moneymaker-expanded-history-v23-65`.

### Viability and timing diagnostics

| Month | Viability AUC | Actual positive anchors | Predicted viable rate | 15m global rank Spearman | Episode-median rank Spearman |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 0.5956 | 24.63% | 0.54% | 0.1465 | 0.2655 |
| 2026-02 | 0.6723 | 22.15% | 0.00% | 0.1331 | 0.2507 |
| 2026-03 | 0.6336 | 24.72% | 0.00% | 0.1470 | 0.2446 |

The episode model therefore learned meaningful out-of-month ordering in all
three months. The within-episode timing signal also remained positive and
strong, with positive episode-correlation rates of 87.5%, 86.3%, and 87.1%.

However, chronological Platt calibration correctly kept the mean predicted
positive probability near the roughly 22-25% base rate. No February or March
anchor reached the fixed semantic 0.50 viability threshold.

### Trading result

| Month | Policy | Evaluated trades | Gross mean | BASE mean | Day-balanced BASE |
|---|---|---:|---:|---:|---:|
| Jan | earliest eligible | 2050 | -0.061% | -1.511% | -1.584% |
| Jan | earliest viable | 11 | +0.312% | -0.817% | -0.497% |
| Jan | viability + rank | 10 | +0.120% | -1.020% | -0.723% |
| Feb | earliest eligible | 1756 | -0.383% | -1.880% | -1.965% |
| Feb | earliest viable | 0 | n/a | n/a | n/a |
| Feb | viability + rank | 0 | n/a | n/a | n/a |
| Mar | earliest eligible | 2035 | -0.048% | -1.576% | -1.613% |
| Mar | earliest viable | 0 | n/a | n/a | n/a |
| Mar | viability + rank | 0 | n/a | n/a | n/a |

In January, timing rank delayed or changed some viable entries but reduced
common-episode realized BASE return by about 0.55 percentage points versus the
first viable anchor. Thus a globally useful timing rank is not automatically
useful inside the tiny high-viability tail.

### Position ledger

For the primary viability-rank policy:

- January BASE: 10 attempts/accepted/closed, complete accounting, marked return
  -1.0199%;
- February: zero attempts, complete accounting, 0.0% marked return;
- March: zero attempts, complete accounting, 0.0% marked return.

Zero-trade months are not profitable evidence and cannot satisfy the promotion
rule.

The broad earliest-eligible policy remained strongly negative and often ended
with unresolved capital because thousands of attempts contend for the fixed
synthetic account. This does not alter the primary-policy failure.

### Decision

Reject exact v2.3. April 2026 and later remain sealed.

The mandatory >=15 monthly trades, positive monthly BASE/day-balanced returns,
positive bootstrap and positive BASE ledger-return criteria all failed.

Do not lower the 0.50 probability threshold or rank gate on these results.

## Diagnostic implication

The new information is sharper than a generic "model is bad" conclusion:

1. episode positive-return probability is predictably ordered out of month;
2. within-episode entry quality is predictably ordered out of month;
3. probability alone is not enough to establish positive expected value because
   payoff magnitude is asymmetric;
4. in January, even the highest calibrated-probability anchors had positive
   gross mean but negative BASE mean;
5. timing rank did not rescue that selected tail.

The next separately frozen branch should therefore estimate the payoff
distribution explicitly, decomposing:

- P(BASE return > 0),
- conditional positive-return magnitude,
- conditional loss magnitude.

A hurdle-style expected-value estimate can then test whether the model can
identify episodes whose favorable payoff asymmetry, not merely win probability,
covers the existing BASE friction. This is a new target decomposition, not a
post-hoc lowering of the v2.3 gate.
