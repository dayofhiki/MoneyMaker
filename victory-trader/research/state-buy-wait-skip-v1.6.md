# State BUY-WAIT-SKIP Policy v1.6 — pre-registration

## Status

Pre-registered after v1.5 failed and before implementing or viewing any v1.6
trading result.

Use only the already-seen January-March 2026 development panels. Do not access a
fresh validation month.

This branch keeps the fixed timing geometry from v1.5 but adds NO-TRADE/SKIP as
an explicit learned action.

## Frozen information set

Use exactly the frozen v1.3-v1.5 causal information set:

- existing point-in-time price/volume/breadth/sequence state;
- exact 8 lagged FINRA short-volume features;
- exact 13 prior-8-K features;
- exact 8 publication-safe FINRA short-interest features.

No feature-family, threshold, interaction, or external source may be selected
using January-March outcomes.

## Fixed action geometry

- BUY means enter now and hold exactly 10 minutes;
- WAIT means wait exactly 5 minutes and reconsider;
- SKIP means terminate the ticker-day with no trade;
- decision checkpoints are offsets 0, +5, and +10 minutes from the first
  eligible state;
- at most one attempted BUY per ticker-day;
- WAIT at +10 terminates as SKIP;
- a missing required exact checkpoint terminates as SKIP.

The first eligible state is selected using only the existing causal eligibility
rule.

## One-step three-action training label

For an eligible state t with valid realized 10-minute base-net return at t and
an exact eligible t+5 state with valid realized 10-minute base-net return:

- BUY value = realized 10-minute base-net return from t;
- WAIT value = realized 10-minute base-net return from t+5;
- SKIP value = 0%.

Assign the label deterministically:

1. SKIP if both BUY and WAIT values are <= 0;
2. BUY if BUY value is > 0 and BUY value >= WAIT value;
3. WAIT otherwise.

This is a fixed one-step action comparison. It does not take a maximum over an
arbitrary future window and does not use future-state features.

Future realized returns are labels only.

Rows without both realized BUY and exact +5-minute WAIT labels are excluded from
model fitting only. Label availability must never define live eligibility.

## Model

Fit one multiclass HistGradientBoostingClassifier to {BUY, WAIT, SKIP}:

- loss = log loss;
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=31;
- min_samples_leaf=300;
- l2_regularization=2.0;
- random_state=20260926;
- existing chronological 80/20 train partition retained;
- existing three-minute training cadence retained.

No class-probability threshold is fitted or tuned. The executable action is the
class with the largest predicted probability.

## Policies

### earliest_eligible_10m_cap1

Primary timing-naive comparator: BUY 10m at the first eligible state.

### buy_wait_skip_cap1

Process exact checkpoints chronologically.

- predicted BUY: attempt BUY immediately and hold 10m;
- predicted WAIT: advance exactly 5m;
- predicted SKIP: terminate with no trade;
- WAIT at +10: terminate with no trade;
- missing next checkpoint: terminate with no trade.

A BUY decision consumes the ticker-day attempt even if unfillable or its
realized evaluation return is unavailable.

No direct-EV gate, rank threshold, probability threshold, retrospective peak,
or outcome-selected fall-through is allowed.

## Diagnostics

For every holdout month report:

- label class prevalence;
- predicted action prevalence;
- multiclass balanced accuracy;
- multiclass log loss;
- confusion matrix / per-class recall;
- trade count and days;
- gross/base/stress mean;
- day-balanced base mean;
- median, p05, positive rate, severe-loss rate, worst day;
- decision-step mix and skipped ticker-days;
- common-episode return delta versus earliest_eligible_10m_cap1;
- external-family coverage;
- pooled trading-day-cluster bootstrap.

## Promotion rule

The exact branch passes development only if ALL conditions hold:

1. at least 15 trades in every month;
2. arithmetic base-net mean > 0 in every month;
3. day-balanced base-net mean > 0 in every month;
4. day-balanced base-net mean strictly above earliest_eligible_10m_cap1 in
   every month;
5. base p05 and stress-net mean are not worse than
   earliest_eligible_10m_cap1 in any month;
6. holdout multiclass balanced accuracy > 1/3 in every month;
7. short-volume latest-prior coverage >=90%, publication-safe short-interest
   latest coverage >=90%, and 8-K query completion =100% in every month;
8. pooled trading-day-cluster bootstrap 95% lower bound > 0.

If all eight pass, freeze the exact branch and pre-register one unseen
validation month before accessing it.

If the branch fails, do not tune action thresholds, timing geometry, class
weights, model capacity, or features on January-March. The next branch must use
an explicit fitted sequential value / Bellman-style formulation rather than
another hand-written action label.

No fresh validation month is consumed here.
