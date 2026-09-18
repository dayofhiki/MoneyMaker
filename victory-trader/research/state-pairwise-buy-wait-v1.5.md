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
