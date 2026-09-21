# Expanded-history anchor hurdle EV v2.4 — pre-registration

## Status

Pre-registered after v2.3 was completed and retired, before implementation and
before viewing any v2.4 output.

Use only September 2025 through March 2026 development data. April 2026 and
later remain sealed.

This branch tests one new target decomposition: estimate the after-cost expected
value of the first eligible 15-minute BUY by separately modeling win
probability, conditional win magnitude, and conditional loss magnitude.

## Motivation fixed before results

v2.3 established two repeatable out-of-month ordering signals:

- first-state episode viability AUC was 0.5956 / 0.6723 / 0.6336 in
  January-March;
- 15-minute within-episode rank Spearman was 0.1465 / 0.1331 / 0.1470.

Yet a calibrated win-probability threshold of 0.50 produced only 12 viable
January episodes and none in February/March; the January selected BASE result
was still negative. The broad first-entry population also had negative gross or
near-zero gross means and strongly negative BASE means.

Therefore P(return > 0) alone is not the economic objective. A lower win
probability can still have positive expectation when wins are larger than
losses, while a higher win probability can lose money when losses dominate.

This branch keeps the fixed 15-minute anchor geometry and explicitly estimates
that payoff asymmetry. It does not tune the failed v2.3 probability gate.

## Frozen data and information set

Development months:

- 2025-09
- 2025-10
- 2025-11
- 2025-12
- 2026-01
- 2026-02
- 2026-03

For every fitted head use the exact multi-source causal feature frame used by
v2.3:

- causal price/volume/breadth state;
- causal ticker-local sequence features;
- current and previous absolute-price transforms;
- lagged FINRA short-volume features;
- strictly-prior 8-K features;
- publication-safe FINRA short-interest features.

No NBBO feature is allowed because the configured historical quote REST and
Flat File paths returned HTTP 403.

No feature subset search, hand interactions, new external source, news text,
cost change, fresh validation month, brokerage action, or threshold tuning.

## Strict past-only folds

Evaluate only:

- January 2026: train/calibrate on Sep-Dec 2025;
- February 2026: train/calibrate on Sep 2025-Jan 2026;
- March 2026: train/calibrate on Sep 2025-Feb 2026.

For each fold sort strictly-prior scoreable training days and assign the first
80% to fit and final 20% to chronological calibration.

Persist provenance proving
max(training/calibration day) < min(evaluation day).

## Fixed episode anchor and return

One observation per ticker-day.

The anchor is the first chronological state satisfying the existing causal
state-eligibility rule. Future label availability must never select a different
anchor.

Economic target R is the existing realized 15-minute BASE-net return from that
anchor.

Anchors with missing R are excluded from model fitting/calibration only and
remain explicit in evaluation/attempt ledgers.

## Hurdle decomposition

For every labeled anchor define:

- W = 1 if R > 0, else 0;
- positive magnitude G = R for W=1;
- loss magnitude L = -R for W=0. Zero return therefore has L=0.

The modeled expected BASE-net return is:

    hurdle_EV = P(W=1) * E[G | W=1, X]
                - (1 - P(W=1)) * E[L | W=0, X]

No additional transaction-cost adjustment is applied because R already includes
the frozen BASE cost scenario.

## Head 1 — win probability

Use the exact v2.3 probability model:

- HistGradientBoostingClassifier;
- loss="log_loss";
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=15;
- min_samples_leaf=75;
- l2_regularization=2.0;
- random_state=20261011;
- no class weights.

Chronological Platt calibration is unchanged in form:

1. score labeled calibration anchors;
2. clip raw probabilities to [1e-6, 1-1e-6];
3. fit one unpenalized logistic regression on raw log-odds;
4. require at least 100 labeled calibration anchors and both classes.

No probability threshold participates in the primary policy.

## Heads 2 and 3 — conditional payoff magnitudes

Fit separate conditional-mean regressors on fit anchors.

### Win magnitude head

Training rows: R > 0.
Target: G = R.

### Loss magnitude head

Training rows: R <= 0.
Target: L = -R.

Both use:

- HistGradientBoostingRegressor;
- loss="squared_error";
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=15;
- min_samples_leaf=50;
- l2_regularization=2.0;
- random_state=20261012 for win and 20261013 for loss;
- train-only target winsorization at 0.5th and 99.5th percentiles.

The raw target is magnitude in percentage points, not log magnitude, because
the hurdle formula requires conditional arithmetic mean payoff.

### Magnitude calibration

For each head, use only matching-sign labeled anchors from the chronological
calibration period.

Require:

- at least 75 calibration anchors for that head;
- at least 5 distinct calibration trading days.

Compute one additive mean-bias correction:

    offset = mean(actual magnitude - raw predicted magnitude)

Apply the fixed offset to all future predictions and clamp the resulting
magnitude at zero.

No slope, selected-tail correction, quantile threshold, or post-result
recalibration.

## Executable policies

All policies use a fixed 15-minute BUY and at most one attempted BUY per
ticker-day.

### earliest_eligible_15m_cap1

Attempt BUY at every ticker-day's first eligible state.

### probability_half_15m_cap1

Diagnostic reproduction of the v2.3 anchor-only semantic gate:

- attempt BUY iff calibrated P(W=1) >= 0.50.

This comparator is diagnostic only and cannot define v2.4 success.

### hurdle_ev_15m_cap1 — primary

At the anchor compute calibrated win probability, calibrated conditional win
magnitude, calibrated conditional loss magnitude, and hurdle_EV.

- attempt BUY iff hurdle_EV > 0 percentage points;
- otherwise SKIP.

The zero threshold is semantic. It must not be tuned.

The first BUY decision consumes the ticker-day attempt even if the entry
reference or future evaluation labels are missing. Do not fall through to a
later state.

No within-episode timing rank is used in the primary policy. v2.3 showed that
the globally useful rank signal can worsen the tiny selected episode tail. The
purpose of v2.4 is to establish a cost-positive episode-selection process
before reintroducing timing freedom.

## Execution and accounting

Keep existing light/base/stress scenarios unchanged.

Persist every BUY attempt and mutually exclusive evaluation reason.

Replay all three policies through the frozen position ledger with unchanged:

- independent synthetic account per evaluation month/scenario;
- $10,000 accounting unit;
- $1,000 fractional order unit;
- no leverage or daily cash reset;
- delayed exits, unresolved capital, same-ticker blocks, missing references
  and session handling.

These remain research accounting conventions, not capital recommendations or
claims of actual fills.

## Required diagnostics

For every evaluation month report:

- date provenance;
- anchor rows and labeled-anchor coverage;
- actual positive rate;
- win-probability AUC, Brier score, log loss and calibration summaries;
- win-magnitude calibration rows/days, raw prediction mean, calibrated
  prediction mean, actual mean and Spearman on positive holdout anchors;
- loss-magnitude equivalent diagnostics on nonpositive holdout anchors;
- hurdle_EV mean, selected rate, holdout Spearman versus realized BASE return
  across all labeled anchors;
- selected predicted hurdle_EV mean versus selected realized BASE mean;
- attempts/evaluated/unevaluable and missing reasons;
- gross/base/stress return distributions;
- day-balanced BASE mean, median, p05, severe-loss rate, worst day;
- common-episode comparison against earliest eligible and probability-half;
- external-data coverage;
- pooled trading-day-cluster bootstrap for hurdle_ev_15m_cap1;
- position-ledger light/base/stress marked return and complete-accounting status.

## Development promotion rule

Do not access April 2026 or any later month unless ALL conditions hold for
hurdle_ev_15m_cap1:

1. at least 15 evaluated trades in January, February and March;
2. arithmetic BASE-net mean > 0 in every month;
3. day-balanced BASE-net mean > 0 in every month;
4. day-balanced BASE-net mean strictly exceeds earliest_eligible_15m_cap1 in
   every month;
5. BASE p05 and STRESS mean are not worse than earliest_eligible_15m_cap1 in
   any month;
6. win-probability AUC > 0.50 and hurdle_EV Spearman versus realized BASE > 0
   in every month;
7. pooled trading-day-cluster bootstrap 95% lower bound > 0;
8. short-volume latest-prior coverage >=90%, publication-safe short-interest
   latest coverage >=90%, and 8-K query completion =100% in every month;
9. BASE-scenario marked account return > 0 and complete_accounting=true in
   every month.

A sparse positive-looking tail fails. Zero trades fail. A light-cost profit
cannot substitute for BASE. Missing outcomes cannot be assigned zero.

Conditional win/loss magnitude Spearman is diagnostic rather than a separate
promotion gate because the primary economic object is their probability-weighted
combination.

## Failure rule

If v2.4 fails, do not tune the zero EV gate, winsorization, 80/20 split,
probability/magnitude head capacity, magnitude calibration, 15-minute horizon,
cost scenarios or feature subset on January-March.

Use the frozen diagnostics to decide the next intervention:

- if hurdle_EV does not rank realized BASE positively, redesign the candidate
  universe or add genuinely new state information;
- if hurdle_EV ranks well and selected gross alpha is positive but BASE remains
  negative, execution-cost observability/protocol is the bottleneck;
- if a cost-positive entry process replicates, only then pre-register timing
  rank and later HOLD/SELL freedom.

No fresh validation month is consumed here.
