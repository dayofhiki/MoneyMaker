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


## Result

Workflow request 44 completed successfully. The exact pre-registered
BUY-WAIT-SKIP classifier failed development promotion.

### Holdout action prediction

The true one-step action labels were heavily SKIP-dominated:

- January: BUY 12.89%, WAIT 12.26%, SKIP 74.85%;
- February: BUY 11.32%, WAIT 10.68%, SKIP 78.01%;
- March: BUY 12.00%, WAIT 11.36%, SKIP 76.63%.

The classifier collapsed almost completely to SKIP:

- predicted SKIP rate: 99.76%, 99.72%, 99.86%;
- predicted BUY rate: 0.22%, 0.26%, 0.12%;
- predicted WAIT rate: approximately 0.02% in every month.

Balanced accuracy was only 0.3350, 0.3356, and 0.3345. Although each value is
numerically above 1/3, that criterion is not substantively informative here:
BUY and WAIT recall were near zero while SKIP recall was approximately 99.9%.

### Trading result

| Month | Trades | Base mean | Day-balanced | p05 | Stress mean |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 18 | -0.904% | -0.930% | -7.632% | -2.828% |
| 2026-02 | 26 | -1.503% | -0.918% | -7.507% | -3.416% |
| 2026-03 | 17 | -1.412% | -2.544% | -9.756% | -3.336% |

The pooled day-balanced mean was -1.350%, with 95% day-cluster bootstrap
interval [-2.899%, +0.442%].

Every executed policy trade occurred at the initial checkpoint. There were zero
BUYs at +5 or +10 minutes. Therefore the branch did not learn a useful
sequential WAIT policy; it mostly learned a dominant SKIP classifier plus a tiny
set of immediate BUY predictions.

### Pre-registered decision

Passed mechanically:

- monthly trade-count sufficiency;
- balanced accuracy > 1/3 every month;
- external-data coverage.

Failed:

- positive arithmetic return every month;
- positive day-balanced return every month;
- improvement over the earliest-entry comparator every month;
- tail/stress protection every month;
- positive pooled bootstrap lower bound.

## Interpretation

Adding SKIP solved the forced-trade problem only superficially. A single
multiclass classifier turns the economic problem into a highly imbalanced class
prediction task. The dominant SKIP class overwhelms BUY and WAIT, while the
relative timing structure found in v1.5 is barely used.

The next formulation should therefore model action values rather than action
labels. In particular, WAIT should inherit the value of a learned future policy
instead of being treated as a flat class.

## Decision

Retire exact v1.6.

Do not tune class weights, class thresholds, checkpoint geometry, model
capacity, or feature subsets on January-March.

Proceed to a pre-registered fitted sequential action-value / Bellman-style
branch with BUY, WAIT, and SKIP represented by values. Use cross-fitted
continuation-policy targets inside the development folds to avoid training a
WAIT target on an in-sample future-policy choice.

No fresh validation month was consumed.
