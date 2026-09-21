# Expanded-history honest stopping distillation v4.1 — pre-registration

## Status

Pre-registered after v4.0 January and February evaluation outputs made the
v4.0 development promotion rule impossible to satisfy, before inspecting the
March v4.0 economic output and before implementing or inspecting any v4.1
output.

April 2026 and later remain sealed.

v4.1 changes one modeling assumption: the downstream policy value used as a
training target must be realized on a chronologically later sample that the
teacher stopping models did not train on. It keeps the v4.0 information set,
entry gate, costs, action boundary, model family, per-minute structure and
30-minute cap unchanged.

## Motivation fixed before results

v3.9 established that the frozen entries contain substantial cost-positive
30-minute exit opportunity and that the v3.7 transition state predicts
remaining option value out of month.

v4.0 attempted direct fit-sample backward induction. January and February
showed a clear optimism failure:

- fit-sample early-minute HOLD-advantage targets averaged roughly +1.5 to
  +1.6 percentage points;
- fitted early-minute HOLD rates were roughly 95%-97%;
- calibration trajectories, which were not used to change v4.0, were already
  BASE-negative;
- evaluation median holding times moved from v3.8's one minute to roughly
  26-28 minutes;
- January/February v4.0 day-balanced BASE returns were about -1.91% and
  -1.83%, both worse than v3.8.

Thus the intervention is not a threshold increase or a shorter forced horizon.
The suspected failure is self-referential optimism: a model trained on a set is
also used to decide which realized downstream payoff from that same set becomes
the earlier-minute Bellman target.

v4.1 removes that same-sample target selection.

## Frozen upstream information

Reuse exactly:

- v3.2 first-anchor panels from request 84;
- frozen 2025 and 2026 state panels already used by v3.7-v4.0;
- v3.3 opportunity model and strict P(opportunity) > 0.5 entry gate;
- v3.7 causal path state;
- all ten position-path variables;
- exact 1/2/3/5/8/13-minute transition deltas;
- no timestamp compression;
- LIGHT/BASE/STRESS execution assumptions;
- hard 30-minute maximum hold;
- strict chronological evaluation folds.

No feature, lag, entry threshold, cost assumption, phase subset, held-minute
subset or maximum horizon changes in v4.1.

## Stage A — teacher fitted stopping on fit only

Train the exact v4.0 minute-specific stopping system on the fold's fit
partition only.

This teacher is intentionally unchanged, including:

- minutes 29 down to 1;
- HistGradientBoostingRegressor;
- squared-error loss;
- learning_rate 0.05;
- 180 iterations;
- 15 leaves;
- min_samples_leaf 75;
- L2 = 2.0;
- fit-only 0.5%/99.5% target winsorization;
- semantic HOLD boundary at predicted advantage > 0.

The teacher is not the v4.1 executable policy. Its role is to define a fixed
downstream behavior that can be evaluated honestly on the later calibration
partition.

## Stage B — honest downstream values on chronological calibration

Freeze the teacher, then score every calibration decision row without fitting
on any calibration outcome.

For every calibration ticker-day, walk backward from minute 29 to minute 1 and
compute the **realized teacher-policy value**:

- if the frozen teacher says EXIT at a state, value is actual BASE return from
  EXIT now;
- if the teacher says HOLD, value is the actual realized value of following the
  teacher at the exact next state;
- minute 29 HOLD resolves at the frozen 30-minute forced exit;
- missing exact next state/open follows the existing first-subsequent-positive-
  open gap rule;
- if no later open exists, the downstream value is unavailable.

Then, independently of the teacher's current action, define for each calibration
state:

    honest_hold_advantage_m =
        actual value of forcing HOLD for one minute
        then following the teacher downstream
        - actual BASE EXIT-now value

The teacher never trained on these calibration outcomes. Therefore the target
contains the teacher's out-of-sample realized continuation value rather than a
same-sample selected value.

## Stage C — student stopping models

Train one student model for each minute 29..1 using only calibration rows with a
valid honest_hold_advantage target.

Use the exact v4.0 model family and capacity:

- squared-error HGBR;
- learning_rate 0.05;
- max_iter 180;
- max_leaf_nodes 15;
- min_samples_leaf 75;
- L2 = 2.0;
- random_state = 20261047 + minute.

Winsorize each minute's student target at its calibration-training 0.5th and
99.5th percentiles.

Require at least 150 valid student target rows per minute. Failure to meet this
is an implementation/data failure, not permission to merge minutes or lower
the requirement after results.

No additional prediction calibration, offset, slope, threshold search or
student Bellman recursion is allowed.

The student target is fixed teacher continuation, so the student never uses its
own in-sample action to manufacture an earlier training label.

## Executable v4.1 policy

On the evaluation month:

1. enter only frozen v3.3 gate-positive anchors at the exact next-minute open;
2. after each completed holding minute score the corresponding student model;
3. HOLD iff predicted honest downstream advantage > 0;
4. otherwise EXIT at the current executable open;
5. repeat only along reached states;
6. force exit at 30 minutes;
7. apply unchanged gap/missing behavior.

The semantic zero boundary is frozen and may not be tuned.

## Frozen comparators

On the same gated entries reproduce:

1. v3.8 one-step recurrent policy;
2. v4.0 same-sample fitted stopping policy;
3. always-HOLD-to-30m.

Primary comparison remains v3.8. v4.0 is included to verify that honest target
construction corrects the identified long-hold optimism rather than merely
changing outcomes arbitrarily.

## Required diagnostics

For every January-March evaluation fold report:

- gated attempts, valid entries, completed/unresolved positions;
- gross/LIGHT/BASE/STRESS arithmetic returns;
- day-balanced BASE;
- median, p05, positive rate, severe-loss rate;
- holding-time distribution, cap reach rate and EXIT reasons;
- teacher calibration realized policy summary;
- per-minute honest target count, mean, winsor bounds, student prediction mean
  and student predicted HOLD rate;
- matched v4.1-minus-v3.8 and v4.1-minus-v4.0 BASE differences;
- trading-day cluster bootstraps using 10,000 samples and seed 20261047;
- account replay under unchanged audit conventions;
- strict provenance.

## Frozen development promotion rule

v4.1 passes only if every January, February and March fold satisfies all of:

1. at least 100 completed trades;
2. exact valid-entry coverage >= 95%;
3. completion coverage >= 90% of valid entries;
4. arithmetic BASE mean > 0;
5. day-balanced BASE mean > 0;
6. v4.1 BASE trading-day bootstrap 95% lower bound > 0;
7. day-balanced BASE strictly exceeds v3.8;
8. matched v4.1-minus-v3.8 day-balanced difference > 0 with bootstrap
   95% lower bound > 0;
9. day-balanced BASE strictly exceeds v4.0;
10. STRESS arithmetic mean is no worse than v3.8;
11. BASE account marked return > 0 with zero unresolved accepted positions.

All conditions must hold in all three months before any sealed validation is
opened.

## Failure interpretation

If v4.1 robustly improves v3.8/v4.0 but remains BASE-negative, stopping has
improved enough that the next branch should add a causal entry
value/abstention model trained on realized v4.1 policy return.

If v4.1 does not robustly improve v3.8, then the current transition state plus
teacher-distilled stopping target is insufficient for executable stopping. The
next intervention must change the causal intraday information or stopping
target family, not tune zero thresholds or the 30-minute cap.

April 2026+ remains sealed after any development failure.
