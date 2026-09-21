# Expanded-history opportunity-gated shared action value v3.3 — pre-registration

## Status

Frozen after v3.2 request 84 completed and before inspecting any v3.3 model
output. Use only September 2025 through March 2026 development data. April 2026
and later remain sealed.

## Motivation fixed before results

v3.2 disproved the idea that replacing fixed 15-minute exits with six
independent EV heads and choosing their maximum is sufficient.

- BASE mean remained negative in January, February and March.
- 1-2 minute direction and EV ordering were informative, but exact best-horizon
  selection was only 17.3%, 22.2% and 22.9%.
- The chosen horizon lost 4.08, 4.00 and 3.42 percentage points versus a
  non-executable hindsight oracle.
- 30 minutes won the max operation 39.7%, 40.0% and 58.6% of the time without
  superior discrimination, consistent with cross-head scale/variance bias.
- 68% of selected states had at least one BASE-positive horizon, so opportunity
  detection and action choice must be separated.

The intervention is therefore structural: one opportunity/abstention head, one
shared action-value scale, calibration after action maximization, and
decision-time action feasibility.

## Frozen data and causal features

Reuse the exact v3.2 multi-horizon anchors from request 84. Preserve the v3.0
filing-semantic + split + supply + multi-source causal feature frame and the
strict chronological folds used in v2.6-v3.2.

No April 2026+ data, new external source, feature search, cost change, horizon
change or post-result threshold selection is allowed.

## Decision-time action feasibility

An action horizon h is available only when its scheduled bar exists before the
known regular-session close. With zero-indexed regular-session minute m, require
`m <= 389 - h`.

This mask uses only the clock known at the decision. Missing future labels,
halts and later missing bars must never be used to remove an action. A selected
action with no evaluable outcome remains an explicit failed/unknown attempt and
has no fallback.

## Opportunity gate

The fit/calibration label is whether any clock-feasible action among
1/2/5/10/15/30 minutes has BASE return above zero. Missing labels are excluded
from label construction but never become inference-time feasibility inputs.

Fit one HistGradientBoostingClassifier on the frozen state features with the
v3.2 classifier capacity and regularization. Calibrate its probability on the
chronological calibration partition with Platt scaling. The frozen gate is
`P(any feasible BASE-positive action) > 0.5`.

## Shared action-value model

Stack every clock-feasible labeled `(state, horizon)` pair. Use one
HistGradientBoostingRegressor with a common state feature frame plus frozen
action encodings:

- horizon in minutes;
- log(1 + horizon);
- one-hot indicator for each frozen horizon.

Use squared-error loss to target expected BASE return, 0.5%/99.5% target
winsorization, learning rate 0.05, 220 iterations, 15 leaves, minimum leaf size
75, L2 regularization 2.0 and seed 20261032.

At inference, score every clock-feasible horizon on this shared scale and choose
the maximum raw Q.

## Policy-level max-selection correction

Individual-action calibration is insufficient because the policy selects a
maximum. On the chronological calibration partition, execute the complete raw
policy: gate above 0.5 and maximum raw shared Q above zero. For its chosen action
compute daily mean optimism `predicted Q - realized BASE`.

Subtract

`max(0, mean daily optimism + 1.645 * daily-mean standard error)`

from the selected maximum Q. Fewer than five evaluated selected calibration
days assigns infinite correction and disables the shared policies. This rule is
frozen before evaluation.

## Policies

1. `feasible_earliest_15m_cap1`: broad fixed-15 comparator.
2. `filing_semantic_split_supply_hurdle_ev_15m_cap1`: reproduced v3.0 fixed-15
   comparator.
3. `adaptive_filing_semantic_hurdle_ev_cap1`: reproduced v3.2 independent-head
   max-EV comparator.
4. `policy_calibrated_shared_q_cap1`: buy the selected shared-Q horizon iff its
   corrected Q is positive; diagnostic isolation of the gate.
5. `opportunity_gated_shared_q_cap1`: primary; buy iff the opportunity gate is
   above 0.5 and selected corrected shared Q is positive.

All policies consume the first selected attempt. No missing-label fallback,
later entry, or substitute horizon is allowed.

## Required diagnostics

For each January-March evaluation month report strict provenance; policy-level
calibration size/days/optimism/correction; opportunity AUC, Brier and log-loss;
attempt/evaluation reasons; GROSS/BASE/STRESS distributions; day-balanced BASE;
action mix and coverage; exact-best rate, oracle regret and any-positive-action
rate; comparator overlap; causal feature coverage; pooled day bootstrap; and
frozen position-ledger BASE/LIGHT/STRESS accounting.

## Development promotion rule

Do not access April 2026 or later unless the primary satisfies all of:

1. at least 15 evaluated trades in each month;
2. arithmetic and day-balanced BASE mean above zero in every month;
3. day-balanced BASE strictly exceeds both fixed-15 and v3.2 in every month;
4. BASE p05 and STRESS mean are not worse than v3.2 in any month;
5. pooled trading-day bootstrap 95% lower bound above zero;
6. selected-action evaluation coverage at least 90% in every month;
7. opportunity AUC above 0.5 in every month;
8. exact-best-horizon rate exceeds random 1/6 in every month and mean oracle
   regret is below v3.2 in every month;
9. causal coverage remains valid;
10. BASE position-ledger marked return is positive with complete accounting in
    every month.

## Failure interpretation

- A useful gate with unstable action ranking points next to richer causal
  continuation/exhaustion state and then short-step HOLD/EXIT learning.
- Better ranking but negative corrected tails points to entry value/calibration
  or unobserved execution cost.
- Positive 1-2 minute actions that fail at longer planned horizons supports the
  final recurrent structure: ENTRY, observe again after one or two minutes,
  then HOLD or EXIT.

Do not tune this exact branch after results.

## Diagnostic-only amendment after request 86

Request 86 completed before this amendment. The frozen primary selected zero
trades in January and February and one trade in March because the policy-level
correction was 4.359, 4.193 and 2.302 percentage points. The opportunity gate
remained informative with monthly AUC 0.645, 0.698 and 0.673.

The original output recorded action regret only for trades surviving the frozen
correction, so it could not distinguish weak shared action ranking from a
conservative policy correction. A diagnostic-only rerun will additionally
record the unchanged shared-Q chosen action over four pre-existing subsets:

- all clock-feasible states;
- gate probability above the frozen 0.5 threshold;
- raw maximum shared Q above the frozen zero threshold;
- both frozen conditions together.

For each subset report chosen-outcome coverage, chosen BASE mean, feasible
hindsight-oracle mean/regret, exact-best rate, any-positive-action rate,
raw-Q/chosen-return Spearman, selected-Q optimism and action mix. This amendment
does not change a model, fit/calibration split, feature, action, threshold,
correction, trade, promotion rule or April seal. Its output is descriptive only
and cannot retroactively promote v3.3.

## Result

Requests 86 and 87 completed successfully. Request 86 ran the frozen policy;
request 87 reproduced it exactly and added only the diagnostic amendment.

Runs:

- https://github.com/dayofhiki/MoneyMaker/actions/runs/35600153954
- https://github.com/dayofhiki/MoneyMaker/actions/runs/35600997146

### Frozen primary policy

| Month | Evaluated trades | BASE mean | Day-balanced BASE | BASE account return | Accounting |
|---|---:|---:|---:|---:|---|
| 2026-01 | 0 | n/a | n/a | 0.000% | complete, zero trades |
| 2026-02 | 0 | n/a | n/a | 0.000% | complete, zero trades |
| 2026-03 | 1 | +3.870% | +3.870% | +0.387% | complete |

This fails the frozen minimum-trade, three-month profitability, bootstrap and
replication criteria. Zero trades are not evidence of profitability. April 2026
and later remain sealed.

### Opportunity gate

| Month | AUC | Brier | Gate-positive rate |
|---|---:|---:|---:|
| 2026-01 | 0.645 | 0.235 | 40.5% |
| 2026-02 | 0.698 | 0.220 | 42.4% |
| 2026-03 | 0.673 | 0.224 | 44.3% |

The gate replicated above-random opportunity discrimination in every month.
This is the successful component of v3.3 and should be retained as a candidate
state representation, not yet as a profitable entry rule.

### Policy-level correction

| Month | Selected calibration rows | Days | Mean daily optimism | Frozen correction |
|---|---:|---:|---:|---:|
| 2026-01 | 16 | 10 | +2.882% | 4.359% |
| 2026-02 | 19 | 13 | +1.884% | 4.193% |
| 2026-03 | 31 | 14 | +1.101% | 2.302% |

The complete max-selection policy was strongly optimistic in calibration. The
frozen upper-confidence correction therefore removed every evaluation trade in
January and February and all but one in March.

### Raw shared action ranking

For the unchanged subset satisfying both frozen pre-correction conditions
(`gate > 0.5` and `max raw shared Q > 0`):

| Month | States / evaluated | Chosen BASE | Feasible oracle BASE | Regret | Exact-best | Any positive action | Q/return Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-01 | 30 / 28 | -1.812% | +3.452% | 5.264% | 21.4% | 71.4% | +0.193 |
| 2026-02 | 30 / 27 | -1.686% | +2.126% | 3.813% | 7.4% | 77.8% | -0.205 |
| 2026-03 | 17 / 16 | -0.237% | +3.117% | 3.355% | 18.8% | 87.5% | +0.347 |

Thus the zero-trade outcome is not merely an overly conservative correction.
Before correction, shared-Q selected tails were BASE-negative in all three
months and action ordering was unstable. The model often detected a state with
some profitable path but chose the wrong planned duration.

Across all clock-feasible states the shared model also chose one minute 66.5%
to 77.8% of the time and never chose two minutes. This removed v3.2's 30-minute
max-scale bias but replaced it with a short-horizon bias rather than learning a
stable relative action policy.

## Decision

Fail v3.3 development promotion. Retain the opportunity gate finding; reject
both independent-head max EV and shared planned-horizon Q as trading policies.

The next experiment should not tune the correction or horizon set. Test whether
new causal state observed after entry contains one-step continuation information:

1. enter conceptually at the frozen first anchor;
2. after one minute, use the then-observable state to compare executable EXIT
   now versus HOLD for one additional minute;
3. model the incremental same-position advantage, treating entry cost as sunk
   and retaining only future exit friction;
4. repeat the diagnostic by time-since-entry and market phase;
5. only if HOLD/EXIT ordering replicates across January-March should it be
   composed into a recurrent one-minute policy with a forced maximum hold,
   halt-aware accounting and no label fallback.

This is the direct bridge from opportunity detection to the intended adaptive
trader: ENTRY, observe new evidence, then repeatedly HOLD or EXIT.
