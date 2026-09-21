# Expanded-history final-scale calibrated hurdle EV v2.5 — pre-registration

## Status

Pre-registered after v2.4 completed and was retired, before implementation and
before viewing any v2.5 output.

Use only September 2025 through March 2026 development data. April 2026 and
later remain sealed.

## Motivation fixed before results

v2.4 produced positive out-of-month ordering in every development month:

- hurdle-EV Spearman versus realized 15-minute BASE return:
  0.1892 / 0.2620 / 0.2109 for January-March;
- win-probability AUC:
  0.5956 / 0.6723 / 0.6293;
- both conditional magnitude heads also had positive holdout Spearman in every
  month.

But the raw hurdle_EV > 0 tail was sparse and over-optimistic. The selected
predicted EV means were +0.467%, +0.507%, and +0.276%, while selected realized
BASE means were -0.320%, -2.288%, and -0.519%.

This branch therefore changes only the calibration of the final combined EV
scale. It does not change the candidate universe, features, component model
capacity, costs, horizon, or semantic zero expected-value decision rule.

## Frozen data and feature information

Development months:

- 2025-09
- 2025-10
- 2025-11
- 2025-12
- 2026-01
- 2026-02
- 2026-03

Use the exact v2.4 multi-source causal feature frame and the exact v2.4 first
eligible ticker-day anchor.

No April 2026+ data, NBBO, feature subset search, hand interaction search, new
external source, cost change, brokerage action, live deployment, or timing
freedom is allowed.

## Strict past-only folds

Evaluate only:

- January 2026 using Sep-Dec 2025 history;
- February 2026 using Sep 2025-Jan 2026 history;
- March 2026 using Sep 2025-Feb 2026 history.

As in v2.4, sort strictly-prior scoreable trading days and allocate the first
80% to model fitting and the final 20% to chronological calibration.

New in v2.5: split the 20% calibration days chronologically into two disjoint
blocks:

1. component calibration block: earlier half of calibration days;
2. final-scale calibration block: later half of calibration days.

When the number of calibration days is odd, the earlier block receives the
smaller half and the final-scale block receives the larger half.

Persist all date provenance and prove all training/calibration dates precede the
evaluation month.

## Frozen component heads

Fit the exact v2.4 models on fit anchors.

### Win probability

HistGradientBoostingClassifier with exactly:

- loss="log_loss"
- learning_rate=0.05
- max_iter=180
- max_leaf_nodes=15
- min_samples_leaf=75
- l2_regularization=2.0
- random_state=20261011

Fit chronological Platt calibration only on the component-calibration block.
Require at least 100 labeled anchors and both classes.

### Conditional win magnitude

Train only R > 0 anchors with the exact v2.4 regressor:

- squared error
- learning_rate=0.05
- max_iter=180
- max_leaf_nodes=15
- min_samples_leaf=50
- l2_regularization=2.0
- random_state=20261012
- train-only 0.5/99.5 percentile target winsorization

Calibrate with one additive mean-bias offset using only positive anchors in the
component-calibration block. Require at least 75 rows and 5 trading days.

### Conditional loss magnitude

Same as the win magnitude head, using R <= 0 and target -R, random_state
20261013, and the same component-calibration requirements.

Compute raw_hurdle_EV exactly as v2.4:

    P(win) * E[gain | win, X]
    - (1 - P(win)) * E[loss | loss, X]

## Final-scale calibration

Score every labeled first eligible anchor in the disjoint final-scale
calibration block using the already frozen component heads.

Fit one ordinary least-squares affine mapping:

    calibrated_EV = intercept + slope * raw_hurdle_EV

Target is the realized 15-minute BASE-net return in percentage points.

Requirements:

- at least 250 labeled final-scale calibration anchors;
- at least 5 final-scale calibration trading days;
- fitted slope must be strictly positive.

If the slope is <= 0 or requirements are not met, the primary policy is disabled
for that fold. Do not flip the sign, force a positive slope, change the split,
or fall back to raw EV.

This affine layer is the only new learned object in v2.5. It preserves the raw
score ordering whenever the slope is positive and moves only its economic
scale/zero crossing.

## Executable policies

All policies use the same fixed 15-minute BUY at the first eligible ticker-day
anchor and at most one attempt per ticker-day.

### earliest_eligible_15m_cap1

Attempt every anchor.

### raw_hurdle_ev_15m_cap1

Diagnostic comparator under the v2.5 nested calibration split:

- BUY iff raw_hurdle_EV > 0.

This is not the exact v2.4 reproduction because component calibration now uses
only the earlier calibration block.

### calibrated_hurdle_ev_15m_cap1 — primary

- if final-scale calibration is valid, BUY iff calibrated_EV > 0;
- otherwise no BUY attempts.

The zero threshold is semantic and must not be tuned.

No timing rank, HOLD, or SELL freedom is allowed.

## Required diagnostics

For each evaluation month report:

- fit, component-calibration, final-calibration and evaluation date provenance;
- component calibration row/day counts;
- final calibration rows/days, slope, intercept, raw-EV Spearman and
  calibrated-EV Spearman;
- evaluation win AUC and conditional magnitude Spearman;
- evaluation raw and calibrated EV Spearman versus realized BASE;
- raw/calibrated selected rates;
- selected predicted calibrated EV versus realized gross/BASE/stress;
- attempts/evaluated/unevaluable reasons;
- day-balanced BASE, p05, severe-loss rate and worst day;
- external-data coverage;
- pooled trading-day cluster bootstrap for the primary;
- frozen position-ledger light/base/stress marked returns and accounting
  completeness.

## Development promotion rule

Do not access April 2026 or later unless ALL conditions hold for
calibrated_hurdle_ev_15m_cap1:

1. final-scale slope > 0 in January, February and March;
2. at least 15 evaluated primary trades in every month;
3. arithmetic BASE mean > 0 in every month;
4. day-balanced BASE mean > 0 in every month;
5. day-balanced BASE mean strictly exceeds earliest eligible in every month;
6. BASE p05 and STRESS mean are not worse than earliest eligible in any month;
7. calibrated-EV Spearman versus realized BASE > 0 in every month;
8. pooled day-cluster bootstrap 95% lower bound > 0;
9. frozen external-data coverage thresholds remain satisfied;
10. BASE position-ledger marked return > 0 and complete_accounting=true in every
    month.

A sparse positive-looking tail, zero-trade month, light-only profit, or missing
outcome cannot satisfy promotion.

## Failure rule

If v2.5 fails, do not tune the affine zero crossing, split ratio, model capacity,
costs, horizon, winsorization, or feature subset on January-March.

Interpret failure as follows:

- if evaluation calibrated-EV ordering collapses, the final-scale mapping does
  not transport and this calibration approach is retired;
- if ordering remains positive but the selected tail is still sparse or
  BASE-negative, stop threshold/calibration engineering and move to
  candidate-universe or genuinely new causal state information;
- if gross selection is repeatedly positive but BASE remains negative, treat
  execution-cost observability/protocol as the next bottleneck;
- only after a replicating cost-positive entry process exists may timing,
  HOLD, or SELL freedom be introduced.

No fresh validation month is consumed here.
