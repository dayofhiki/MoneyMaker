# State Pairwise Buy-Wait Preference v1.5 — pre-registration

## Status

Pre-registered after v1.4 failed and before implementing or viewing any v1.5
trading result.

Use only the already-seen January-March 2026 development state panels. Do not
load or inspect a fresh validation month.

This branch deliberately removes the absolute predicted-EV entry gate. It tests
whether the repeatedly observed within-episode relative timing signal can become
an executable trading decision when the learning objective itself is pairwise.

## Frozen information set

Use exactly the v1.3/v1.4 causal information set:

- existing causal price/volume/breadth/sequence state;
- exact 8 lagged FINRA short-volume features;
- exact 13 prior-8-K features;
- exact 8 publication-safe FINRA short-interest features.

No external feature may be added, removed, thresholded, hand-crossed, or selected
using January-March outcomes.

All point-in-time rules from v1.3 remain unchanged.

## Fixed action geometry

To avoid another horizon-selection problem:

- BUY holding period is fixed at 10 minutes;
- WAIT step is fixed at exactly 5 minutes;
- no direct-EV model or +0.50% EV gate participates in the primary policy;
- a ticker-day gets at most three BUY/WAIT decisions, at offsets
  0, +5, and +10 minutes from its first eligible state;
- if all three decisions say WAIT, the ticker-day is skipped;
- at most one attempted BUY per ticker-day.

The first eligible state is the earliest chronological state satisfying the
existing causal state-eligibility rule. No realized label is used to choose it.

A later checkpoint is valid only when a scoreable state exists at exactly the
scheduled timestamp. Missing a required checkpoint ends the path without
fall-through to an outcome-selected later state.

## Pairwise training label

For any eligible state t with an exact eligible state at t+5 minutes:

- NOW value = realized 10-minute base-net return from buying at t;
- WAIT value = realized 10-minute base-net return from buying at t+5;
- pairwise label = 1 when NOW value > WAIT value, else 0.

Future information is used only to construct this supervised label. The t+5
state's features never enter the feature vector at t.

Rows lacking either valid 10-minute realized return are excluded from model
fitting only. Label availability must never affect live signal eligibility.

## Pairwise model

Fit one HistGradientBoostingClassifier:

- loss: log loss;
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=31;
- min_samples_leaf=300;
- l2_regularization=2.0;
- random_state=20260925;
- chronological 80/20 fit/calibration partition retained for diagnostics, but no
  outcome-fitted probability threshold is allowed;
- use the existing three-minute training cadence.

The classifier predicts P(NOW beats WAIT5).

No probability calibration or threshold fitting is performed on
January-March. The semantic decision threshold is fixed at 0.50.

## Policies

### earliest_eligible_10m_cap1

At the first eligible state in each ticker-day, attempt a 10-minute BUY.
One attempt per ticker-day.

This is the primary timing-naive comparator.

### fixed_wait5_10m_cap1

Diagnostic only. If the exact +5-minute checkpoint exists, attempt a 10-minute
BUY there.

### fixed_wait10_10m_cap1

Diagnostic only. If exact +5 and +10-minute checkpoints exist, attempt a
10-minute BUY at +10.

### pairwise_wait_cap1

Starting from the first eligible state:

1. at offset 0, BUY if predicted P(NOW > WAIT5) >= 0.50; otherwise WAIT;
2. if WAIT, move to the exact +5-minute checkpoint and repeat;
3. if still WAIT, move to the exact +10-minute checkpoint and repeat;
4. if the +10-minute decision also says WAIT, SKIP the ticker-day;
5. when BUY is chosen, hold exactly 10 minutes;
6. the first BUY decision consumes the attempt even if the selected entry is
   unfillable or its realized evaluation label is missing.

There is no absolute-return gate, rank quantile, tuned probability cutoff,
future maximum, or retrospective local-peak selection.

## Evaluation

Leave one month out and train on the other two.

Report:

- classifier rows, NOW-positive rate, ROC AUC, balanced accuracy, and log loss
  on each holdout month;
- trade count, trading days, gross/base/stress mean;
- day-balanced base mean, median, p05, positive rate, severe-loss rate,
  worst day;
- entry-delay distribution and decision-step mix;
- common-episode return delta versus earliest_eligible_10m_cap1;
- fixed +5 and +10 minute timing baselines as diagnostics;
- pooled trading-day-cluster bootstrap for pairwise_wait_cap1;
- external-family scoreable coverage.

The historical earliest_ev_cap1 result may be printed for context only. It is
not a tuning target.

## Promotion rule

The branch passes development only if ALL conditions hold:

1. pairwise_wait_cap1 has at least 15 trades in every month;
2. arithmetic base-net mean is positive in every month;
3. day-balanced base-net mean is positive in every month;
4. day-balanced base-net mean is strictly above earliest_eligible_10m_cap1 in
   every month;
5. base p05 and stress-net mean are not worse than
   earliest_eligible_10m_cap1 in any month;
6. holdout pairwise ROC AUC is > 0.50 in every month;
7. short-volume latest-prior coverage >=90%, publication-safe short-interest
   latest coverage >=90%, and 8-K query completion =100% in every month;
8. pooled trading-day-cluster bootstrap 95% lower bound for pairwise_wait_cap1
   day-balanced base-net return is above zero.

If all eight pass, freeze the exact branch and pre-register exactly one unseen
validation month before accessing it.

If this branch fails, do not tune the 5-minute wait, 10-minute holding period,
three-decision cap, 0.50 probability threshold, classifier capacity, or feature
subset on January-March. The next branch must move to a more explicit sequential
policy/value formulation rather than hand-tuning this pairwise rule.

No fresh validation month may be consumed by this experiment.


## Result

Workflow request 43 completed successfully. The exact pre-registered pairwise
BUY/WAIT branch failed development promotion.

### Pairwise prediction

The pairwise classifier was weak but directionally consistent out of month:

| Month | Labeled rows | ROC AUC | Balanced accuracy | NOW positive rate |
|---|---:|---:|---:|---:|
| 2026-01 | 318,668 | 0.5083 | 0.5048 | 49.80% |
| 2026-02 | 255,925 | 0.5106 | 0.5080 | 49.70% |
| 2026-03 | 293,393 | 0.5108 | 0.5072 | 49.66% |

Thus the relative five-minute timing comparison was learnable slightly better
than chance in every month, but only by a small margin.

### Trading result

| Month | Policy | Trades | Gross mean | Base mean | Day-balanced | p05 | Stress mean |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | earliest eligible 10m | 2,097 | +0.114% | -1.339% | -1.348% | -6.721% | -3.505% |
| 2026-01 | pairwise wait | 1,696 | +0.101% | -1.343% | -1.354% | -6.490% | -3.500% |
| 2026-02 | earliest eligible 10m | 1,776 | -0.122% | -1.624% | -1.683% | -7.370% | -3.824% |
| 2026-02 | pairwise wait | 1,646 | -0.064% | -1.556% | -1.584% | -7.062% | -3.752% |
| 2026-03 | earliest eligible 10m | 2,057 | -0.018% | -1.563% | -1.554% | -6.694% | -3.797% |
| 2026-03 | pairwise wait | 1,756 | +0.084% | -1.439% | -1.444% | -6.356% | -3.661% |

The pooled pairwise day-balanced mean was -1.458%, with 95% day-cluster
bootstrap interval [-1.579%, -1.339%].

### Timing diagnostic

Among common fillable episodes, the pairwise policy improved realized base-net
return versus the first eligible entry by:

- January: +0.166 percentage points;
- February: +0.187 percentage points;
- March: +0.186 percentage points.

Most common episodes nevertheless entered at the same checkpoint:

- January: 82.6% same, 11.1% delayed 5m, 6.3% delayed 10m;
- February: 77.0% same, 19.4% delayed 5m, 3.6% delayed 10m;
- March: 81.5% same, 13.8% delayed 5m, 4.6% delayed 10m.

The policy also skipped baseline-only episodes whose realized baseline means
were -0.753%, -0.300%, and -1.275% in January-March respectively. However,
episodes fillable only after a pairwise BUY decision were also negative.

### Pre-registered decision

Passed:

- monthly trade-count sufficiency;
- pairwise AUC > 0.50 every month;
- tail/stress protection versus earliest eligible entry;
- external-data coverage.

Failed:

- positive arithmetic return every month;
- positive day-balanced return every month;
- day-balanced improvement versus the comparator every month (January missed
  narrowly);
- positive pooled bootstrap lower bound.

## Interpretation

The experiment confirms that the repeated relative-timing signal is real but
small in economic magnitude under this formulation. Moving entries according
to the learned five-minute preference improved common-episode returns by roughly
0.17-0.19 percentage points, while broad 10-minute runner entries had near-zero
gross mean and roughly 1.4-1.5 percentage points of modeled execution drag.

The key failure is therefore not just choosing NOW versus WAIT. The policy still
trades the less-bad action when both available actions have negative value.
It lacks an explicit NO-TRADE/SKIP action.

## Decision

Retire the exact v1.5 pairwise BUY/WAIT branch.

Do not tune the wait interval, hold horizon, checkpoint count, probability
threshold, model capacity, or feature subset on January-March.

The next branch may keep the same fixed 5-minute/10-minute geometry and frozen
information set, but must formulate BUY, WAIT, and SKIP jointly as explicit
actions. No fresh validation month was consumed.
