# State One-Step Stop v0.5 — pre-registration

## Status

Pre-registered before implementation and before viewing any v0.5 result.

This experiment uses only the already-seen January-March 2026 development
panels. It must not load or inspect a fresh validation month.

## Motivation

State Wait Value v0.4 learned its future-maximum target out of month, but the
target itself was structurally biased positive because it compared one current
state with the maximum over many states in a 15-minute future window. About 85%
of labeled states truly had a better future opportunity and the semantic zero
stopping threshold produced no trades.

The next experiment removes the max-over-future operation entirely.

## Fixed information set

Use the same causal feature family as v0.4:

- point-in-time state features;
- contemporaneous leave-one-out runner breadth;
- the existing 36 causal ticker-local lag features;
- current and previous absolute price transforms.

No news, future bars as features, fresh month, severe-loss classifier, dynamic
stop branch, or new execution assumption is allowed.

## Buy action model

Retain the existing calibrated direct base-net EV regressors for 5-, 10-, and
15-minute actions.

The executable buy gate remains fixed at predicted best base-net EV >= 0.50%.

## One-step continuation targets

Fit one continuation-delta model for each action horizon h in {5, 10, 15}.

For a state at minute t, define:

    delta_h(t) = realized_base_net_h(t + 1 minute)
                 - realized_base_net_h(t)

A label exists only if:

1. t and t+1 are in the same trading-day/ticker episode;
2. the next state timestamp is exactly one minute later;
3. both current and next realized base-net labels for horizon h are valid.

There is no maximum across future timestamps and no maximum across action
horizons in the continuation label.

Future returns are used only to construct supervised labels. No t+1 state
feature may enter the model input at t.

## Models

For each 5-, 10-, and 15-minute continuation target:

- histogram gradient-boosting regressor;
- existing chronological train-only fit/calibration split;
- existing three-minute training-state cadence;
- fixed causal feature set above;
- target winsorization at the 0.5th and 99.5th fit-sample percentiles;
- train-only affine calibration with the same non-inverting constraints as the
  direct EV model.

## Causal policies

### earliest_ev_cap1

Enter the first chronological state in a ticker-day whose best predicted direct
action EV is at least 0.50%. Choose the highest predicted-EV action horizon.
Permit at most one attempted entry per ticker-day.

### one_step_stop_cap1

At each chronological state:

1. choose the horizon with the highest predicted direct base-net EV;
2. require that best direct EV >= 0.50%;
3. inspect the continuation-delta prediction for that same chosen horizon;
4. enter if predicted next-minute continuation delta <= 0.00 percentage points;
5. otherwise wait one minute and re-evaluate causally;
6. permit at most one attempted entry per ticker-day.

The 0.00 threshold is semantic and fixed: enter when the chosen action is not
expected to improve by delaying its entry exactly one minute.

If the first accepted signal is unfillable or lacks the selected realized label,
the attempted entry is consumed. The policy must not fall through to a later
state using future label availability.

## Evaluation

Leave one month out, train on the other two, and repeat for January, February,
and March.

Report:

- trades, days, and ticker-day episodes;
- base, gross, and stress mean return;
- day-balanced base mean;
- p05, severe-loss rate, and worst day;
- entry minute and action-horizon mix;
- holdout continuation-delta diagnostics by horizon;
- diagnostics on states where that horizon is the direct-EV-selected action and
  the 0.50% buy gate is satisfied;
- common-episode timing and realized-return deltas versus earliest_ev_cap1;
- pooled day-cluster bootstrap interval for one_step_stop_cap1.

## Success rule

Do not promote or consume a fresh month unless one_step_stop_cap1:

1. has at least 15 trades in every month;
2. has positive base-net mean in every month;
3. has positive day-balanced base-net mean in every month;
4. improves day-balanced base-net mean over earliest_ev_cap1 in every month;
5. does not worsen p05 or stress-net mean versus earliest_ev_cap1 in any month;
6. has positive holdout continuation-delta Spearman in every month when pooled
   across EV-qualified chosen-action states; and
7. has a pooled trading-day-cluster bootstrap 95% lower bound above zero.

If this exact branch fails, do not tune the zero threshold or change the
one-minute delay on January-March. Retire the branch or change the objective /
information set.


## Result

Workflow run: https://github.com/dayofhiki/MoneyMaker/actions/runs/35353031154  
Artifact: `moneymaker-state-one-step-stop-v05-28`

The exact pre-registered branch failed.

### Policy outcome

| Month | Baseline base mean | One-step base mean | Baseline day-balanced | One-step day-balanced |
|---|---:|---:|---:|---:|
| 2026-01 | -2.737% | -2.977% | -2.521% | -2.665% |
| 2026-02 | +2.833% | +1.241% | +4.673% | +1.095% |
| 2026-03 | +0.910% | +0.849% | -0.203% | -0.320% |

The one-step policy retained adequate sample size (18, 32, and 20 trades) but
did not improve any month on the pre-registered day-balanced criterion. Pooled
day-balanced base mean was -0.305% with a trading-day bootstrap 95% interval of
[-3.480%, +2.599%].

### Continuation diagnostic

The one-minute delta target was nearly unpredictable over the full holdout state
population. Spearman correlations by action horizon were only about 0.006 to
0.013. In the EV-selected chosen-action pool the pooled Spearman remained
positive in each month (0.091, 0.037, 0.072), but the signal was too weak to
produce useful stopping behavior.

The stopping policy usually entered at the same minute as the baseline. Among
common episodes, the mean base-net delta versus baseline was -0.240%, -0.478%,
and +0.009% for January, February, and March respectively.

## Decision

- Retire the exact one-step continuation branch.
- Do not tune the zero threshold or one-minute delay on January-March.
- Do not consume a fresh validation month.
- Retain the contrast between this weak local-delta signal and the materially
  stronger episode-normalized rank signal from v0.3. The useful timing
  information appears to live on a broader within-episode relative scale rather
  than in the next-minute derivative.
- The next branch should reuse the episode-rank model itself but replace its
  sparse absolute top-25% gate with a threshold-free causal stopping rule based
  on the rank score trajectory.
