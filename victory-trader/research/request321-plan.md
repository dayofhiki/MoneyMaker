# R321: nearby evolution versus elapsed opportunity drift

Register this plan and input pins remotely BEFORE new real-data fits or local-pair
evaluation. Parent R320 closeout93a78761020c4cffe5a7fdd1bc2d452b1fdc1b5c/Draft
PR93. This follows its recorded R321 proposal, not a search over new windows.
Routine research publication and cached execution have standing user authorization.

## Frozen provenance and support

Only official R319 run37588633963/artifact11467966003 and official R320
run37590161637/artifact11467839297. The nine files read are SHA256 pinned in
request321-inputs.json. Keep9325 original training states/8792 complete states,
336 complete training episodes,50 mixed training episodes/all47825 pairs and
total fitting mass50. Keep828 evaluation identities/19530 observed states/18793
complete states. Copy all saved features, labels, missingness, clocks and scores.
No acquisition, label/stop/cost changes, sealed June HOLD or July-August, model
promotion or live trading. Known May development reuse is not fresh validation.

## One new elapsed-only model

Fit ONE arm A on the existing three R319 HISTORY variables only:
trajectory_elapsed_s, trajectory_log_observations, trajectory_lag_age_s, including
their existing missing flags. Slice their exact per-feature transform parameters
from official R320 W (which copies official R319 T); reproduce the subset using
all complete original training states and independent weights within1e-9.
Do not normalize on pairs or evaluation states.

Use the SAME saved positive-negative pair manifest, both orientations at half
pair weight, equal day/mixed-episode/pair mass, LogisticRegression(C=.1,
l1_ratio=0.,solver=lbfgs,fit_intercept=False,max_iter=2000,tol=1e-8,
random_state=20265405). Raw current-state decision_function only; no probability
or returns claim. Preserve frozen U/V/W/Z/T/W0 scores; V remains a secondary
ablation and is never selected as a fallback. A0 broadcasts A's own first score.
Chronological A uses only preceding dates and the corresponding frozen R320 W
fold transform; no-pair fits are explicitly unscored, with all held rows retained.

## Two fixed evaluation questions

Anchor: original R320 whole-episode within-episode AUROC and pooled/same-day
between-episode ranks, preserving original complete-state weights. Replay saved
R320 metrics within1e-9. All scores, including censored states, remain persisted.

PRIMARY diagnostic: enumerate EVERY positive-negative COMPLETE evaluation pair
within the SAME original episode with absolute clock difference <=30000ms.
No future-best state, phase selection, earliest-success restriction or pair cap.
This30s proximity is inherited from R319's lag; no alternate proximity search.
Each eligible pair has equal weight within episode, each eligible episode equal
mass within its day, and each eligible day equal mass. Report the weighted
fraction with positive score > negative score (ties half), called local_pair_rank,
not an unconditional state AUROC. Persist all pairs and an828-identity episode
ledger, including zero-pair identities. Evaluation eligibility is outcome
conditional by definition and cannot be used for online admission.

Bootstrap1000 whole-day/ticker-day multiplicities on ORIGINAL weights, seeds
20265550/20265551. Never treat pairs as independent observations. Report95%
intervals and valid draws for W-A, W-U, W-V, W-Z, W-T, A-A0, W-W0 and V-A in
both whole-episode timing and local_pair_rank. Fixed-fit intervals omit fit
uncertainty. Descriptive per-day ranks and whether positive endpoints came
earlier/later are reported, not selected.

Nine gates: >=30 mixed TRAINING episodes; >=30 local-pair EVALUATION episodes;
>=4 local evaluable dates; W local rank > A, U and.5; W-A improves on a majority
of local evaluable dates; separate DAY and TICKER-DAY positive lower local-rank
CI for W-A and W-U. Keep all failed gates; no replacing W with V or changing
thresholds after seeing results. Pair counts are availability, not independent
sample size. Passing diagnoses a reused-development price/activity signal beyond
this elapsed control; it does not isolate all temporal confounding or justify
trading/promotion. A failed diagnostic redirects the next separate preregistered
model toward training objective/time confounding rather than static interaction.

Persist coefficient/transform/weight audits, full scores, pairs and ledger, fold
scores and intervals. Synthetic tests cover exact proximity, censoring, ties,
episode boundaries, uniform eligible mass, whole-cluster resampling, transform
subset/replay and current-state inference. Full suite and critical Ruff; one
official cached run after local validation; compare JSON/parquet values, score
ordering and missingness within1e-9, then retire its one-shot workflow. Do not
change main or research-request.json.
