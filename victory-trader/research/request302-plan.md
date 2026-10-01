# R302 preregistration — bounded one-stage HOLD learning and policy consistency

Status: implemented for one consolidated research dispatch. 49 tests and research
lint passed; full local execution exactly reproduced R301 and all saved outer
pi1 predictions, with all86 positions resolved. The local economic gate failed;
no policy/feature/seed/threshold changes followed that result. The design below
was fixed before R302 results. May5-8 only; June HOLD and later dates sealed.

## Hypothesis and arms

R301 fixed pre-entry chart availability without improving profits. Test whether
the second policy-improvement stage helps or destabilizes decisions. No new
features, thresholds, model complexity, minimum hold or extra iterations.

A: exact R301 B replay (pi2, two-stage nested improvement).
B PRIMARY: pi1 trained on the fixed old10m/2% trailing reference policy's WAIT
advantages. Fit on the three other development days and test the excluded day.
Runtime re-evaluates pi1 at every observed completed second, HOLD iff score>0.
Use R301's already-fixed outer-pi1 seeds 20263975 + fold*100, enabling exact
prediction reproduction against saved pi1_prediction_pct. This is not a new
seed search. The model is fit on the SAME 28 full-history feature columns,
reachable states, weighting and hyperparameters as R301.

The pi1 target follows pi0 AFTER the next observation, while deployed pi1
subsequently follows itself. Likewise pi2 was learned against inner pi1 targets
and subsequently follows pi2. Explicitly audit these differences; neither is a
proof of policy convergence or safe policy improvement. B's simpler fixed target
does not become an optimal target merely because it is fixed.

## Frozen boundary

Same 86 entries / 97 candidates, frozen R296 entry gates, -2.5% BASE-net stop,
costs, caps, absorbing pending next-regular-open exits and full causal history.
No entry filters or removal of nine cost-only stops/eleven no-HOLD trajectories.
Dependencies R301 run36831962753 and R298 run36801235949; no API acquisition.
No dates beyond May5-8, no June model selection, no post-result parameter tuning.

## Learning and diagnostic audit

- B fits only TARGET_PI0 labels on the outer three days. Validate excluded-day
  provenance and use `_fit` and fixed seed unchanged. Reproduce saved outer pi1
  predictions within1e-9 before replaying B; fail integrity if mismatched.
- A reproduces original submissions, reasons and returns within1e-9.
- Report all-entry economics for frozen pi0, A/pi2 and B/pi1. Compare B-A with
  day and ticker-day cluster intervals, plus paired B-old. No production fallback
  to whichever reference wins. Main question is the incremental second stage.
- Rank/calibrate both predictions against identical frozen-pi0 and saved outer
  pi1 target labels, as all-reachable/first/actually visited state diagnostics.
- Construct post-hoc diagnostic WAIT-follow-deployed-policy labels: for A use
  its pi2 predictions at downstream states; for B use its pi1 predictions. These
  outer-day future labels are NEVER training targets or model/threshold selectors.
  Risk/cap/pending execution remain exactly as `continuation_targets` specifies.
- For A compare saved pi1-follow diagnostic target with self-pi2 target; for B
  compare learned pi0 target with self-pi1 target. Report sign disagreements,
  mean absolute differences, prediction calibration and ranked correlations.
  Examine first and actually visited states separately from unused counterfactuals.
- This audit can find policy mismatch, not prove that it alone causes loss or that
  a diagnostic oracle action is learnable. No retrospective peak labels.

## Preregistered gate and reporting

Primary B must reconcile all86 and resolve>=95%, reproduce A and saved pi1,
have positive BASE mean and>=3/4 positive days, beat A and old trailing on paired
means, have positive mean excluding largest winner, and positive weighted rank
against its LEARNED frozen-pi0 advantage. Same sign threshold0 throughout.

Report BASE/LIGHT/STRESS, median, CVaR5, concentration, daily means, holding times,
forced/model exits, first-decision exits, clock feature availability, gross/cost
drag, risk-compatible and whole-path upside, and execution gaps. Four reused
days and upstream entry OOF not nested prohibit promotion/untouched claims.

## Tests and next boundary

Test held-out day labels cannot alter its fitted model; supplied feature sets,
seeds and training day exclusion reach `_fit`; self-policy targets match a fixed
reference when actions coincide and change only when downstream actions change;
pending forced risk remains absorbing. Retain inherited future-feature/session/
history tests and strict JSON, exact replay, all-entry integrity checks.

If B fails, do not continue adding policy iterations or select a diagnostic
threshold. Use the audit to decide a separately preregistered model/action-value
or entry/cost experiment. If B passes, freeze for additional later-May development
before considering June. Passing reused development is not proof of profitability.
