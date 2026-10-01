# R302 findings — fixed-reference pi1 does not restore profitability

Official run https://github.com/dayofhiki/MoneyMaker/actions/runs/36833594641.
Source7f731e6c0403e99e7285d1157a2fc659e953f309; artifact11147774149.
Execution successful, economic gate FAILED. Official JSON equals complete local
verification. R301/pi2 and all saved outer pi1 predictions exactly reproduced;
same86 positions resolved. June HOLD remains sealed, no policy promotion.

## Stage comparison

| Metric | A: pi2/R301 | B: pi1/R302 |
|---|---:|---:|
| Mean BASE net % | -1.468004 | -1.449165 |
| Median BASE net % | -1.320558 | -1.203446 |
| Positive entries | 8/86 | 6/86 |
| Median hold s | 14 | 11 |
| First-decision model exits | 44 | 55 |
| CVaR5 % | -5.319604 | -5.125685 |
| Gross mean % | +0.075017 | +0.093839 |

B-A +0.018839pp with day CI[-0.138493,+0.124388], ticker-day
CI[-0.145802,+0.196444]. Weak and uncertain change, not reliable improvement.
B daily BASE means -2.149283,-1.133288,-1.860052,-0.993780%: all negative.
LIGHT -0.613667%, STRESS -3.684646%; largest-winner-excluded -1.523730%.
B-old +0.238829pp on paired means, but ticker-day CI crosses0. Neither learned
stage nor frozen old policy has profitable economics on this population.
Mean cost drag1.543003pp versus gross edge0.093839%; friction is not sufficient
as the only explanation because LIGHT remains negative. These are trade means,
not compounded account returns. 48/86 B fills wait>3s; next prints are not quotes.

## Policy-reference mismatch

B weighted rank against learned frozen-pi0 target +0.094381, but against
WAIT-follow-deployed-pi1 outcomes -0.073074. First-decision ranks +0.171713
versus -0.078652. Learned-reference and self-policy label signs differ at22/75
first states,34/245 actually visited reachable states,4661/10619 all reachable
counterfactual states. The all-state figure is not a fraction of deployed trades.
Reference/self differences are diagnostic, not proof of a sole causal failure.

A also has reference/self mismatch:11/75 first states,79/330 visited states.
A's archived outer pi1 diagnostic labels are not identical to the inner target
policies that fitted pi2. Therefore this audit does not directly reconstruct every
pi2 training residual or prove failure of policy improvement mathematically.

Frozen pi0 imposes a10min timer;64.38% of reachable state rows are at age>=600s,
and66.19% of its targets are exactlyzero. Learned runtime pi1 instead can continue
to the common cap. This expired-reference support is intentional and was
preregistered, not data leakage. It further qualifies interpreting global ranks
as evidence of useful deployed continuation.

## Additional training-calibration inconsistency found during analysis

`hierarchical_crack_entry_controller._fit` trains the tree using episode/day
weights, then computes its additive offset as unweighted `mean(y-prediction)`.
Thus its final mean calibration uses a different empirical distribution from
its optimization and principal weighted diagnostics. This shared behavior was
frozen for R299-R302; no source code was changed during this result analysis.

Refitted the exact four B models with identical targets/features/seeds solely
to inspect training residuals. The weighted residual correction and resulting
weighted mean prediction bias, in percentage points:

| Excluded day | Current unweighted offset | Weighted residual offset | Current weighted bias |
|---|---:|---:|---:|
| May5 | +0.104591 | -0.083182 | +0.187773 |
| May6 | +0.136467 | -0.042682 | +0.179150 |
| May7 | +0.122313 | +0.077943 | +0.044370 |
| May8 | +0.024227 | -0.164159 | +0.188387 |

This is measured on TRAINING folds, not held-out calibration improvement.
An additive shift can change score>0 decisions; ranking within a single fold
is unchanged. It does not prove correcting the offset will make a profitable
model, nor explain every reference/self mismatch. No alternative offset policy
has been replayed or selected from held-out outcomes in this analysis.

## Next research priority

Before more features or policy iterations, preregister an isolated weighted
training-offset comparison against exact R302 B. Keep the same fixed pi0 target,
tree,28 features,seeds,risk,costs and sign threshold0. Compute correction solely
from outer training residuals with the existing day/episode weights, preserving
the rest of target clipping/preprocessing. Implement within the next experiment
without silently rewriting historical shared models. Assert weighted training
mean residual is zero and held-day labels never affect calibration.
Recompute held-out own-policy diagnostics AFTER frozen fit, never feed them back.
Economic gates, all-entry outcomes and common/self-policy distinctions remain
required. A failed gate must not be overridden by rank or training calibration.
Next experiment not launched; June remains sealed.
