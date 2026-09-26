# Request243 findings — calibration FAIL, no development evaluation

Authoritative run: [36259470906](https://github.com/dayofhiki/MoneyMaker/actions/runs/36259470906).
Evaluated source: `fb3bc70f2c6753d314e25939824e136f0fef69f6`.
The workflow, all 12 integrity tests and the separate repository test workflow
completed successfully. This technical success is not a research gate pass.

## What was implemented

Causal ENTER / WAIT / ABSTAIN and HOLD / EXIT policy rollouts with one
chronological policy-improvement pass. The early teacher trains on Apr30–May4;
it generates action targets on May5–8 without fitting on those dates. The final
student is tested on the eight May calibration days with no fitted thresholds
or offsets. No hindsight best future return is a training target.

## Authoritative calibration result

| Policy | Entry attempts | Resolved trades | Unresolved | Resolved trade mean |
|---|---:|---:|---:|---:|
| Student full controller | 173 | 100 | 73 | -1.6464% |
| Earlier teacher | 14 | 7 | 7 | -0.3555% |
| No WAIT, one-shot entry | 119 | 70 | 49 | -1.6535% |
| No HOLD, immediate next-state exit | 173 | 110 | 63 | -1.5008% |
| First-event one-step baseline | 433 | 345 | 88 | -1.2836% |

433 archived candidate episodes. Student used WAIT 442 times and HOLD 1,437
times; using the actions did not establish economic benefit. Its resolved
trades averaged -3.6270% under STRESS costs, and 48% lost at least 2% under BASE.
The table conditions on different resolved subsets, so its means are not a
paired superiority comparison. All full-controller daily primary means are
unknown because each date contains unresolved trades. **There is no valid
aggregate portfolio return or final balance.** Do not interpret the zero count
of known positive days as eight observed losing-day returns.

The fixed calibration gate failed. June23–29 development position shards were
downloaded by Actions but the evaluator did not load or score their outcomes.
No new dates or holdout were opened. No model was promoted.

## What this does and does not establish

This is an implemented sequential decision experiment, not evidence that the
model is converging in profitability. The student learned to act much more
frequently than the earlier teacher, but the resolved selections were negative.
The first chronological teacher is itself weak and sparse; simply using its
continuation as a target cannot guarantee that valuable waits are taught.
Policy improvement also changes the downstream holding policy, so a predicted
teacher continuation is not automatically a calibrated student-policy value.

Two separate bottlenecks remain:

1. **Execution reconstruction.** Source position artifacts contain exact-minute
   next-open references. A missing reference does not prove that a later market
   fill is impossible. Request243 keeps these unresolved instead of using future
   availability as a selection filter or silently pricing them at zero. A next
   experiment should reconstruct pending orders from the raw observed scan/tape,
   with a predeclared expiry/deadline, and distinguish unavailable entry from
   unavailable liquidation. It must not choose a favorable later price or the
   hindsight last observed row.
2. **Action-value optimism / weak teacher.** Even the resolved cases are losing.
   Repairing missing execution will not itself solve this. Compare predicted
   ENTER against realized continuation under the same frozen policy using
   strictly chronological out-of-sample blocks, and inspect how much positive
   WAIT support the teacher actually supplies. Do not tune a new percentile or
   value offset on the already-observed calibration result.

Before another economic claim, isolate these factors with identical causal
features, a predeclared execution contract, and same-population baselines.
The eventual full trader still needs all-HOT admission, reentry, shared capital
and position limits, and raw-tape/quote validation. This experiment inherited
the BUY-gated sample and does not validate any of those components.

## Reproducibility and artifact note

The authoritative run used Request214's pinned NumPy 2.4.6, pandas 3.0.6,
scikit-learn 1.9.1, SciPy 1.17.1, joblib 1.6.0 and threadpoolctl 3.7.0.
The auxiliary local runtime was different and gave different trade counts
(149 attempts, 96 resolved, mean -1.3920%). It is not the authoritative result;
no settings were changed to choose between those outcomes. Input hashes and
runtime versions are in `results/request243.json`.

A subsequent serialization-only repair stores policy fields as dictionaries,
so a normal importer can reconstruct `Policy(**payload['student'])` without a
pickle dependency on the CLI's `__main__.Policy`. It does not change any action,
target, fit, gate or reported result. The original run's joblib object can be
read only with that class explicitly bound when loading; JSON/parquet results
are unaffected.
