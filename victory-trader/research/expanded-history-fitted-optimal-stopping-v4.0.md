# Expanded-history fitted optimal stopping v4.0 — pre-registration

## Status

Frozen after remaining-option observability v3.9 request 97 passed every
pre-registered bridge condition in January, February and March, and before
implementing or inspecting any v4.0 output.

Use only September 2025 through March 2026 development data. April 2026 and
later remain sealed.

v4.0 changes the decision formulation, not the information set. The frozen
v3.3 opportunity gate still decides which anchors become conceptual entries.
The frozen v3.7 causal state and exact transition lags still describe an open
position. The new component is finite-horizon fitted optimal stopping: at each
reached minute estimate whether HOLD followed by the learned downstream policy
has greater expected BASE value than EXIT now.

## Motivation fixed before results

v3.8 showed that recursively composing the one-step sign classifier was too
myopic:

- median hold was one minute in January, February and March;
- BASE trade means were -0.975%, -1.149%, and -1.334%.

v3.9 then established both prerequisites for a less-myopic stopping rule:

1. the same frozen gated entries contained hindsight best-exit day-balanced
   BASE returns of +3.193%, +2.622%, and +3.114%;
2. the unchanged causal path-transition state ranked same-minute residual
   remaining option value positively in every month, with selected-set
   bootstrap lower bounds +0.920%, +0.715%, and +1.153%.

Therefore the next test should learn an executable downstream value recursion
rather than another fixed horizon or another one-step direction label.

## Frozen upstream data and information

Reuse exactly:

- v3.2 first-anchor panels from request 84;
- frozen 2025 state panels from request 54 and frozen 2026 state panels from
  request 8;
- the unchanged v3.3 opportunity model and strict
  P(opportunity) > 0.5 entry gate;
- strict chronological folds used by v3.7-v3.9;
- all v3.7 causal state features;
- all ten v3.5 position-path variables;
- exact 1/2/3/5/8/13-minute deltas for all ten path variables;
- no timestamp compression;
- existing LIGHT/BASE/STRESS execution assumptions;
- hard maximum holding time of 30 minutes.

No feature, lag, entry threshold, cost assumption, held-minute subset, phase
subset or cap may be selected after results.

## Frozen economic scale

Every stopping value is expressed as full round-trip BASE return percentage
from the position's original entry reference.

For a state reached after m completed holding minutes:

- EXIT value = full round-trip BASE return using the original modeled BASE buy
  fill and the currently executable modeled BASE sell fill;
- HOLD value = the realized BASE return obtained by waiting one more minute and
  then following the already-fitted downstream stopping policy.

Because both actions share the same entry fill, subtracting them is equivalent
to comparing future sell value while preserving the economically relevant
full-trade scale.

## Fit-only backward recursion

The stopping models use only the fold's fit partition. The chronological
calibration partition is diagnostic only and may not adjust a threshold,
offset, feature, model capacity or minute subset in v4.0.

Fit one separate model for each completed holding minute 29, 28, ..., 1, in
that order.

### Minute 29

For each fit row at minute 29 with a valid current EXIT reference:

- EXIT value = BASE return from exiting now;
- HOLD downstream value = BASE return from the mandatory 30-minute exit;
- if the exact forced exit open is absent, use the first subsequent observed
  positive open under the frozen gap rule;
- if no later open exists, the HOLD target is unavailable.

Target:

    hold_advantage_29 =
        HOLD downstream BASE return - EXIT-now BASE return

Fit the minute-29 model. On fit rows, define the fitted-policy realized value as
the actual HOLD downstream return when the model predicts advantage > 0,
otherwise the actual EXIT-now return.

### Minutes 28 through 1

For minute m in descending order:

- EXIT value = BASE return from exiting at the current decision open;
- if the exact next decision state m+1 exists and its fitted-policy realized
  value is known, that value is the HOLD downstream target;
- if the next decision state is absent, model decisions stop and HOLD resolves
  at the first subsequent observed positive open at or after the next expected
  decision-open timestamp, matching the frozen v3.8 gap behavior;
- if no such open exists, the HOLD target is unavailable.

Target:

    hold_advantage_m =
        downstream fitted-policy BASE return - EXIT-now BASE return

The downstream realized value used for fit recursion is determined only by
models already fitted at later minutes. No evaluation or calibration outcome
enters a fit target.

This is ordinary fitted backward induction on historical transitions. The
in-sample Bellman bootstrap is explicitly accepted as a development modeling
assumption; chronological calibration and evaluation diagnostics must reveal
whether it generalizes.

## Frozen per-minute model

Each minute model is one HistGradientBoostingRegressor with:

- loss = squared_error;
- learning_rate = 0.05;
- max_iter = 180;
- max_leaf_nodes = 15;
- min_samples_leaf = 75;
- l2_regularization = 2.0;
- random_state = 20261046 + minute.

Use the exact v3.7 transition feature frame.

For each minute independently, winsorize the fit target at its fit-only 0.5th
and 99.5th percentiles. Calibration and evaluation outcomes are never clipped.

No Platt scaling, additive offset or learned threshold is used. The executable
decision boundary is the semantic zero:

- HOLD iff predicted downstream advantage > 0;
- otherwise EXIT.

## Executable evaluation trajectory

For every evaluation anchor above the frozen opportunity gate:

1. attempt entry at the exact anchor+1-minute open;
2. missing/non-positive entry reference is recorded and no fill is invented;
3. after each completed holding minute, use only the state then observable;
4. score the corresponding frozen minute model;
5. HOLD iff predicted fitted downstream BASE advantage > 0;
6. otherwise EXIT at the current exact decision open;
7. after a HOLD, repeat only on the state actually reached;
8. minute 29 may HOLD only to the hard 30-minute cap, after which exit is
   mandatory.

If an expected reached decision state or decision open is missing, stop model
decisions and exit at the first subsequent observed positive open. If none
exists, the accepted position is unresolved.

No state after an executed EXIT may influence the policy.

## Frozen comparators

Reproduce on the same evaluation fold and same gated entries:

1. **v3.8 one-step recurrent comparator**: exact frozen v3.7 classifier composed
   using strict P(HOLD wins) > 0.5;
2. **always-HOLD-to-30m comparator**: same entry, no early model exit.

The primary matched comparison is v4.0 versus v3.8 because it isolates whether
the less-myopic value recursion improves the failed recurrent policy.

## Required diagnostics

For each evaluation month report:

- gated attempts, valid entries, completed and unresolved positions;
- valid-entry and completion coverage;
- EXIT reason counts;
- arithmetic gross/LIGHT/BASE/STRESS return;
- day-balanced BASE return;
- median, p05, positive-trade rate and BASE <= -5% severe-loss rate;
- mean/median/p10/p90/max holding minutes;
- distribution of first EXIT minute and fraction reaching the 30-minute cap;
- number of HOLD decisions;
- matched return differences versus v3.8 and always-HOLD-to-30m;
- 10,000-sample trading-day bootstrap of v4.0 day-balanced BASE return;
- 10,000-sample trading-day bootstrap of matched v4.0-minus-v3.8 BASE
  difference;
- calibration-partition trajectory diagnostics using the already-fitted policy,
  with no adjustment based on them;
- fit target counts and target/prediction summaries for every minute;
- strict fold provenance.

Use deterministic bootstrap seed 20261046.

## Frozen account replay

Replay v4.0 and comparators independently per month/scenario using the existing
MoneyMaker audit conventions:

- $10,000 synthetic initial cash;
- $1,000 fixed fractional order budget;
- no leverage;
- exits before entries at an identical timestamp;
- same-ticker open positions block another entry;
- insufficient free cash blocks an entry;
- deterministic ticker ordering only as an audit tie-break;
- unresolved accepted positions retain committed capital and a last observable
  mark.

Missing entry references are data-coverage misses, not open accounting
liabilities. Therefore v4.0 does **not** reuse the old
`complete_accounting = no missing entries` promotion test. Instead report them
explicitly and require:

- exact valid-entry coverage >= 95%;
- zero unresolved accepted BASE positions.

This definition is frozen now because v3.8 showed that the older flag conflates
a never-opened missing reference with unresolved committed capital.

## Frozen development promotion rule

v4.0 passes only if every January, February and March evaluation fold satisfies
all of:

1. at least 100 completed v4.0 trades;
2. exact valid-entry coverage >= 95%;
3. completed coverage >= 90% of valid entries;
4. arithmetic BASE mean > 0;
5. day-balanced BASE mean > 0;
6. trading-day bootstrap 95% lower bound for v4.0 BASE return > 0;
7. v4.0 day-balanced BASE return strictly exceeds the reproduced v3.8
   recurrent comparator;
8. matched v4.0-minus-v3.8 day-balanced BASE difference > 0 and its
   trading-day bootstrap 95% lower bound > 0;
9. STRESS arithmetic mean is no worse than the reproduced v3.8 comparator;
10. BASE account marked return > 0 with zero unresolved accepted positions.

All ten conditions must hold in all three months.

A positive hindsight oracle, positive calibration result, LIGHT-only profit,
pooled-only profit or sparse tail cannot substitute for a failed evaluation
month.

## Decision after v4.0

If v4.0 passes, freeze the complete entry + fitted-stopping policy and
separately preregister the first untouched April 2026 validation before
accessing April data.

If v4.0 fails:

- if it captures substantially more of the v3.9 oracle ceiling than v3.8 but
  remains BASE-negative, the next branch may add a causal entry-value/abstention
  model trained on realized v4.0 policy return;
- if fitted stopping does not robustly beat v3.8, the Bellman formulation does
  not sufficiently exploit v3.9 observability and the next branch must change
  the value target or add genuinely new causal intraday information;
- do not tune the zero boundary, minute-specific thresholds, 30-minute cap,
  lags or model capacity on January-March.

April remains sealed after any v4.0 development failure.
