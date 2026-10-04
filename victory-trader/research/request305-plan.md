# R305 preregistration — feasible upside versus causal observability

Fixed before the first R305 run. Diagnostic only: do not change or select entry,
HOLD/EXIT policy, stop, cap, execution, costs, thresholds, features or dates.
Dependencies: official R304 run36883479409 and R298 run36801235949. Reproduce
all86 R304 B decisions and returns exactly from saved scores before diagnosis.
May5–8 only; June and later dates sealed; no market acquisition.

For each frozen position enumerate the existing completed-second decision
submissions before the first mandatory BASE-net -2.5% stop or common cap, plus
that terminal submission. Map each to the first regular print open at/after
submission, preserving pending gaps. Stop submission is absorbing; its pending
fill remains allowed, but no later rebound can be chosen. Also enumerate the
same decision schedule without the stop as a separately named whole-path
diagnostic. These maxima use future knowledge: upper bounds, never teachers or
trading profits. Missing terminal fills censor maxima; never label them losers.

Report all86 positions: cost-uncoverable under reachable maxima, feasible
positive returns missed by R304, captured positives, gross and BASE/LIGHT/STRESS
maxima, gross/net +5/+10/+20 support, per-day counts, oracle/model gap and
cost-only stops. Preserve11 positions with no eligible HOLD action in economics.

At each risk-reachable state label suffix maximum net return and its advantage
over EXIT-now using only the allowed current/future submission schedule. Fit
diagnostic prediction models exclusively on the other three days. Labels use
future outcomes; X is exactly R304's28 causal columns, never exit-now labels,
future maxima, raw highs, stop times or decision/fill labels.

PRIMARY observability: separate first-HOLD-snapshot logistic models for net
cost coverage (>0), net >=5/10/20; C=.1, median imputation with missing indicators,
standardization, no class balancing, max_iter2000, seed20264050+fold*100.
Train on eligible first decisions only; report first-ineligible counts separately.
Secondary: identical220-iteration,15-leaf, min-leaf75, l2=3, lr=.04 tree
classifiers on all eligible states and a regression on maximum EXIT advantage
(existing clipped fitter with weighted original-target residual offset). Same
day/episode weights. Single-class training folds use their training prevalence.
All labels, imputation, scaling and offsets exclude the scored day.

Report weighted AUROC/AP/Brier, training-prevalence baseline AP/Brier, per-day
and fixed predicted quartile labels, independently for first decisions,
all counterfactual reachable states, deployed R304 visited states and states
whose observed net return has not already reached the target. Confidence
intervals resample whole day or ticker-day clusters, never independent seconds.
No score cutoff or policy is selected or deployed. Continuous ranks are secondary;
an oracle ceiling alone is not actionable expected value. Four reused days and
upstream entry OOF not nested prohibit promotion or independent-validation claims.

Tests: absorbing stop/rebound and stop-cap tie, long pending fills, missing cap
fill censoring, suffix maxima, immutable causal feature columns, excluded-day
target mutation, single-class fold fallback, rejection of future dates/duplicates,
and inherited reproduction/history/pending-risk tests. Dispatch the same design
once with the consolidated request workflow after local verification.
