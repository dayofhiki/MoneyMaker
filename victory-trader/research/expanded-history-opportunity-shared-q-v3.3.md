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
