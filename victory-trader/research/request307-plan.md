# R307 preregistration — nested monotone probability calibration

R306 run37211254516 improved first-cash net+5 ranking C0.597127 versus
B0.551921 and cost-only D0.527301; all3 two-class held days improved. It failed
the preregistered signal gate because C Brier0.103026 is worse than B0.098203,
D0.100021 and the state-training-prior baseline0.081417. Ticker-day uncertainty
also remains substantial. No R306 promotion or probability/threshold inversion.

Fixed before first R307 fit. Isolate calibration, keeping R306's99 C features,
model settings,97 candidates,4526 states, target, dates and all trading rules.
Dependencies: R306 states/JSON only, with ZERO market acquisitions. June sealed.
Replay all saved C net+5 predictions exactly from three-day fits before use.

For each outer held day, create inner day-OOF C predictions on the other three
days: train on TWO days and score the excluded training day using same99
features/HGB settings and fixed seed20264200+outer*100+inner*10+5. Extract each
episode's FIRST saved cash snapshot only, retaining all training episodes.
Fit a two-parameter monotone sigmoid on logit(clipped inner OOF probability),
with intercept unpenalized, slope>=0, L2 penalty0.5*slope^2 and day/episode
weights normalized to mean1. Use L-BFGS-B and no hyperparameter/threshold search.
Single-class or failed optimizer must use a smoothed training-only prevalence;
record the failure/fallback. A zero slope is a failed discrimination case,
never silently forced positive or replaced after observing outer outcomes.

Apply this fixed training-only map to the outer three-day model's saved score.
Report raw C, calibrated E, cost-only D and a FIRST-SNAPSHOT training prevalence
benchmark, plus the original R306 state-weighted prevalence separately. Calibration
uses first-snapshot outcomes only; later snapshots are secondary transport tests.
No entry/HOLD policy or threshold is selected from calibrated probabilities.

Evaluate first,+20,+60,+120 clock snapshots and all states: AUROC/AP/Brier,
predicted versus observed mean, fixed probability bins, per-day metrics and
paired C/E cluster Brier intervals by day/ticker-day on common first episodes.
Primary gate: exact saved C replay; E Brier improves C,D and first-prior baseline;
E AUROC>=raw C; all folds have positive calibration slopes and equal within-day
rank where both labels exist. A gain from becoming a constant prior does not
establish momentum capture. Sparse9 first-snapshot winners and four reused days
prohibit promotion regardless of gate. Report all failures without retuning.

Tests: held/inner-day exclusion, held-target mutation leaves its parameters/scores
unchanged, monotonic mapping/constant fallback, nonfinite probability rejection,
exact saved-score mismatch failure, original causal allowlist and fixed-snapshot
invariants. Keep pre-entry risk/cost and future-feature tests. Record full R306
findings, acquisition coverage and results along with the R307 request.
