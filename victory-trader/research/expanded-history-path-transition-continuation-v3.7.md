# Expanded-history path-transition persistence continuation v3.7 — pre-registration

## Status

Frozen after v3.6 request 93 completed and before implementing or inspecting any
v3.7 output. Use only September 2025 through March 2026 development data.
April 2026 and later remain sealed.

v3.7 changes the causal state-transition representation only. It does not tune a
holding horizon, probability threshold, action threshold, market phase, feature
subset, execution cost, fold, opportunity gate or model capacity.

## Motivation fixed before results

The path-aware branch now has a consistent pattern:

- v3.5 added ten causal position-relative path features. January improved but
  missed only the bootstrap lower bound; February and March passed all frozen
  conditions.
- v3.6 kept the same state but changed sign classification to arithmetic
  expected-value regression. Point estimates stayed positive in all three
  months, but January and February bootstrap lower bounds crossed zero and only
  March passed completely.

The remaining hypothesis is state aliasing. Two positions can have the same
current entry return, drawdown, rebound or path efficiency while moving in
opposite directions. A trader observes whether the position path is
strengthening, stalling or decaying, not only its current level.

The generic state already has a frozen causal lag sequence. The ten
position-relative v3.5 path variables, however, enter only at their current
value. v3.7 tests whether exact causal changes in those path variables carry the
missing continuation information.

## Frozen data, geometry, label and upstream gate

Inherit unchanged from corrected v3.5:

- exact v3.2 first-anchor panels from request 84;
- frozen 2025 state panels from request 54 and frozen 2026 state panels from
  request 8;
- strict past-only fit/calibration/evaluation day partitions;
- unchanged v3.3 opportunity model and frozen P(opportunity) > 0.5 stratum;
- conceptual entry at the exact next-minute open after the frozen anchor;
- follow-up decisions after 1-29 completed holding minutes;
- EXIT now uses the exact currently observable open and HOLD one more minute
  uses the exact next-minute open;
- no missing-timestamp fallback;
- BASE future sell friction only, with entry cost sunk;
- unchanged one-step arithmetic HOLD advantage label;
- April 2026+ remains sealed.

## The only planned feature change: exact path-transition deltas

Start from the exact v3.5 causal feature frame, including all ten current
position-path variables.

Reuse the already-frozen causal sequence lag grid:

- 1 minute;
- 2 minutes;
- 3 minutes;
- 5 minutes;
- 8 minutes;
- 13 minutes.

For every v3.5 position-path variable and every frozen lag k, add exactly one
transition feature:

    current path value - path value at exact t-k minutes

The ten source path variables are:

1. entry-relative current return;
2. maximum favorable excursion;
3. maximum adverse excursion;
4. drawdown from running high;
5. rebound from running low;
6. running high-to-low range;
7. minutes since running high;
8. minutes since running low;
9. observed-minute fraction;
10. path efficiency.

An exact lag exists only when the same ticker-day position has a row exactly k
minutes earlier. A row-count shift across a missing minute, halt or data gap is
not an exact lag and must produce missing transition features rather than
compressing time.

No lag subset, interaction subset, phase subset or transition threshold may be
selected after results. Current values remain present, so the model can use
either level or causal change.

## Frozen classifier and action rule

Return to the v3.5 sign-classification objective because v3.6 magnitude
regression did not improve monthly robustness.

Use the exact v3.5 model family and capacity:

- HistGradientBoostingClassifier;
- log loss;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- minimum leaf size 75;
- L2 regularization 2.0;
- deterministic seed 20261043;
- chronological Platt calibration on the unchanged calibration partition.

The diagnostic action remains HOLD iff calibrated P(HOLD wins) > 0.5,
otherwise EXIT. The 0.5 boundary is unchanged and may not be tuned.

## Required diagnostics

Repeat the v3.5 diagnostics unchanged for January, February and March:

- all-anchor and frozen opportunity-gate strata;
- exact-reference coverage and missing reasons;
- AUC, Brier and log loss;
- predicted HOLD rate;
- always-HOLD incremental mean;
- classifier incremental mean and day-balanced mean;
- held-minute and market-phase descriptive breakdowns;
- 10,000-sample trading-day cluster bootstrap;
- strict fold provenance.

Additionally report transition-feature exact-lag coverage by lag. This is a
data-quality diagnostic only and cannot be used to select a lag after results.

## Frozen bridge criterion

Proceed to a separately preregistered executable recurrent one-minute
ENTRY -> HOLD/EXIT policy only if the opportunity-gate overall stratum satisfies
all five conditions in every January, February and March fold:

1. AUC > 0.5;
2. classifier incremental mean > 0;
3. day-balanced classifier incremental mean > 0;
4. exact-reference coverage >= 90%;
5. trading-day bootstrap 95% lower bound > 0.

If v3.7 passes, the next experiment may compose the frozen classifier into a
causal recurrent path with a forced 30-minute maximum, explicit halt/missing
reference handling and account-level accounting.

If v3.7 fails, do not tune the lag grid, 0.5 threshold, held-minute buckets,
phase buckets or model capacity on January-March. Treat minute-level
position-path transition state as insufficiently robust and move to a genuinely
new intraday information source or a different transition target. April remains
sealed either way.
