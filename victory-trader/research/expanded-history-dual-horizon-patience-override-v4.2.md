# Expanded-history dual-horizon patience override v4.2 — pre-registration

## Status

Pre-registered after v4.1 request 99 failed to improve v3.8 in January,
February and March, and before implementing or inspecting any v4.2 output.

April 2026 and later remain sealed.

v4.2 preserves v3.8 as the executable short-horizon controller. It adds one
narrow intervention: when v3.8 would EXIT, estimate whether forcing exactly one
additional minute of patience and then returning to the v3.8 controller has
positive BASE value.

The override model may use only the two independently established signals:

1. v3.7 calibrated one-minute HOLD probability;
2. v3.9 predicted excess remaining-option value.

No raw feature expansion, threshold search, horizon search or phase filtering is
allowed.

## Motivation fixed before results

The development sequence now supports three separate facts.

- v3.7 showed reproducible one-minute continuation/decay information.
- v3.8 used that information as an executable controller, but exited too early
  and remained BASE-negative.
- v3.9 showed that medium-horizon remaining opportunity is independently
  observable and often economically large.
- v4.0 overheld because same-sample Bellman bootstrapping amplified optimistic
  downstream values.
- v4.1 removed that optimism and normalized holding time, but still underperformed
  v3.8 in every month because it distilled a poor long-horizon teacher.

Therefore v4.2 does not replace the strongest executable baseline. It treats
v3.8 as the default policy and asks only whether the medium-horizon signal can
rescue a subset of premature v3.8 exits.

## Frozen upstream information

Reuse exactly:

- v3.2 first-anchor panels from request 84;
- frozen 2025 and 2026 state panels already used by v3.7-v4.1;
- v3.3 opportunity model family and strict P(opportunity) > 0.5 entry gate;
- v3.7 causal path state and exact 1/2/3/5/8/13-minute transition deltas;
- v3.7 strict semantic HOLD boundary P(HOLD wins) > 0.5;
- v3.9 fit-minute baseline definition and excess remaining-option target;
- v3.9 semantic positive boundary at predicted excess option value > 0;
- unchanged LIGHT/BASE/STRESS execution scenarios;
- hard 30-minute maximum holding time;
- exact-timestamp/no-gap-compression behavior;
- existing account replay conventions: $10,000 audit starting cash and $1,000
  fixed fractional order budget.

No v3.7/v3.9 threshold, lag, cost or horizon may be changed from prior results.

## Honest override-training split

The fold's chronological calibration trading days are split once, before any
v4.2 outcome is inspected.

- Sort unique calibration trading days ascending.
- Calibration-A = first floor(N/2) days.
- Calibration-B = all remaining days.
- Require at least 5 distinct days in each half.

Calibration-A is used only to calibrate the upstream teacher scores used to
construct honest Calibration-B targets.

Calibration-B outcomes are used only to train the v4.2 patience override.

Evaluation outcomes are never used in either step.

## Stage A — partial-history v3.8 teacher and v3.9 signal

Using the fold fit partition plus Calibration-A only:

- train the unchanged opportunity model on fit and calibrate on Calibration-A;
- train the unchanged v3.7 transition classifier on fit rows and calibrate on
  Calibration-A rows;
- build v3.9 remaining-option labels with the unchanged fit-only held-minute
  baselines;
- train the unchanged v3.9 remaining-value regressor on fit rows and apply its
  additive offset from Calibration-A.

Score all Calibration-B rows with:

- `short_hold_probability`;
- `predicted_excess_option_value_pct`;
- partial-history opportunity probability.

The v3.8 teacher action is frozen:

    HOLD iff short_hold_probability > 0.5
    otherwise EXIT

## Honest one-minute patience target on Calibration-B

For every Calibration-B decision state with a valid current BASE EXIT value,
compute the realized value of the frozen v3.8 teacher backward from minute 29
to minute 1.

For a state where the v3.8 teacher says HOLD, its teacher value is the actual
realized value of following v3.8 at the exact next state.

For a state where the v3.8 teacher says EXIT, its teacher value is the actual
BASE EXIT-now return.

Then for every gate-positive state where the teacher says EXIT, define:

    patience_override_advantage_pct =
        actual value of forcing HOLD for one minute
        then following frozen v3.8 downstream
        - actual BASE EXIT-now return

If the exact next state is unavailable, the existing first-subsequent-positive-
open gap behavior supplies the forced-HOLD downstream value. If no later open
exists, the target is unavailable.

This target answers only the intervention v4.2 will actually make. It does not
ask for an optimal horizon and does not use hindsight-best exit.

## Patience override model

Train one global HistGradientBoostingRegressor on Calibration-B teacher-EXIT
states with a valid patience target.

Inputs are exactly two columns:

- `short_hold_probability`;
- `predicted_excess_option_value_pct`.

No original market/path feature may enter the override model.

Fixed model settings:

- loss = squared_error;
- learning_rate = 0.05;
- max_iter = 120;
- max_leaf_nodes = 7;
- min_samples_leaf = 100;
- l2_regularization = 2.0;
- random_state = 20261048.

Winsorize the Calibration-B target at its 0.5th and 99.5th percentiles before
fitting. Require at least 500 valid teacher-EXIT target rows. No output
calibration, threshold search, score weighting or interaction engineering is
allowed.

## Final evaluation upstream models

After the override model is frozen, reproduce the standard full pre-evaluation
upstream models for the evaluation month exactly as prior research did:

- opportunity model = fit + full chronological calibration;
- v3.7 classifier = fit + full chronological calibration;
- v3.9 remaining-value model = fit with full calibration offset.

The override model itself is not retrained on the full calibration partition,
because Calibration-B contains its honest training outcomes.

## Executable v4.2 policy

On every reached evaluation state:

1. if v3.7 calibrated short HOLD probability > 0.5, HOLD;
2. otherwise score the frozen patience override model;
3. if predicted patience override advantage > 0, HOLD for exactly one more
   minute;
4. otherwise EXIT now;
5. after any HOLD, recompute both signals at the next reached state and apply
   the same rule again;
6. force exit at the unchanged 30-minute cap;
7. preserve the existing delayed-open gap behavior.

The semantic zero override boundary is frozen. A patience override never grants
a multi-minute commitment; it only buys one additional minute before
re-evaluation.

## Frozen comparators

On the exact same evaluation entries reproduce:

1. v3.8 recurrent one-step policy — primary comparator;
2. v4.1 honest stopping distillation — diagnostic comparator;
3. always-HOLD-to-30m — diagnostic comparator.

## Required diagnostics

For each January-March fold report at minimum:

- gated attempts, valid entries, completed and unresolved trades;
- gross/LIGHT/BASE/STRESS return summaries;
- day-balanced BASE and 10,000-sample trading-day bootstrap;
- matched v4.2-minus-v3.8 BASE difference and bootstrap;
- matched v4.2-minus-v4.1 BASE difference;
- mean/median/p10/p90/max holding minutes;
- short-term HOLD count;
- patience-override HOLD count and fraction;
- EXIT reason counts;
- Calibration-B target count, target mean, winsor bounds;
- override prediction mean and positive-prediction rate;
- monthly synthetic account start balance, ending marked balance and return for
  LIGHT/BASE/STRESS;
- account accepted/blocked/closed/unresolved counts;
- strict fold provenance.

From v4.2 onward, every research summary must explicitly state the synthetic
account starting balance, ending balance and marked return.

## Chained January-March audit account

In addition to the three monthly-reset audit accounts, concatenate January,
February and March v4.2 trajectories chronologically and replay one BASE audit
account without resetting cash between months.

Frozen chained-account conventions:

- initial cash = $10,000 on the first January event;
- fixed $1,000 order budget throughout;
- same cash reservation, same-ticker blocking and exit-before-entry ordering;
- no leverage;
- no monthly reset;
- report final marked balance, total marked return and unresolved accepted
  positions after the final March event.

This chained account is an audit diagnostic, not a deployment forecast.

## Frozen development promotion rule

v4.2 passes only if every January, February and March fold satisfies all of:

1. at least 100 completed trades;
2. valid-entry coverage >= 95%;
3. completion coverage >= 90% of valid entries;
4. arithmetic BASE mean > 0;
5. day-balanced BASE mean > 0;
6. v4.2 BASE trading-day bootstrap 95% lower bound > 0;
7. day-balanced BASE strictly exceeds v3.8;
8. matched v4.2-minus-v3.8 day-balanced difference > 0 and its bootstrap
   95% lower bound > 0;
9. STRESS arithmetic mean is no worse than v3.8;
10. monthly BASE audit ending balance > $10,000 with zero unresolved accepted
    positions.

Additionally, the chained January-March BASE audit account must finish above
$10,000 with zero unresolved accepted positions.

All conditions must pass before any sealed April data is opened.

## Failure interpretation

If the patience override robustly improves v3.8 but remains BASE-negative, the
stopping controller is improving but the frozen entry population is too broad
for current execution costs. The next branch should train a causal entry
abstention/value model against realized v4.2 policy returns.

If the patience override does not improve v3.8, the v3.9 medium-horizon score is
useful diagnostically but does not contain enough information to decide an
executable one-minute rescue. The next branch must add genuinely new causal
intraday information or a different policy target rather than tune the zero
boundary, 0.5 boundary or 30-minute cap.

April 2026+ remains sealed after any development failure.
