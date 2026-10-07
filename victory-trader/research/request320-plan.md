# R320: within-episode relative objective under frozen R319 trajectories

Registered before real-data fits or R320 evaluation outcomes. Parent official
R319 closeoutc480d461f716ed6c666a86ca51969fee2b8c09fb/Draft PR92. Source and
synthetic tests were prepared without fitting real states. Routine isolated
publication/cached research is authorized by the user on2026-10-07; existing
sealed dates/costs/stops/no-promotion constraints remain unchanged.

## Frozen inputs

Use official R319 run37588633963, artifact11467966003, source
8084ff22b535ae76e8adf57b0485f6bbe34ec2c6. ZIP digest verifies
23a01bac33e992b35e2dc6fbfaa8eb469cd4119fd2cf006047cb09b75df06cd2.
request320-inputs.json pins all six files read: JSON, all training trajectories,
evaluation states, episode ledger, outcome-independent evaluation manifest,
chronological scored states. No raw acquisition or new clock/label computation.
Keep9,325 original training observations,8,792 complete states/336 episodes,
all828 evaluation identities,19,530 observations/18,793 complete states.
Build no history after censor filtering. No June HOLD/July-August access.

## Pair objective and matched control

Within each COMPLETE training episode, enumerate every positive-negative state
pair for inherited reachable BASE net>=5 labels. All50 mixed episodes and47,825
pairs; no strongest pair, best delay or future-max state selection. Outcomes
select TRAINING pairs by definition; all evaluation states are scored regardless
of outcomes. Persist full pair/episode manifest before fitting. Each endpoint's
features are causal at ITS clock. A later training endpoint is a learning target
comparison, never an inference input to the earlier observation.

Equal day mass among days with mixed episodes, equal mixed-episode mass within
day, and equal pair mass within episode. Effective fitting-weight sum=50, not
47,825. Include both orientations with complementary labels, half weight each.
Copy EXACT official R319 state transforms from its audited JSON (336 complete
training episodes): clipping, missing flags and scaling stay frozen. Verify their
training-only replay within1e-9; do not learn transforms on pair differences or
evaluation states. Unsupervised transform support336 and supervised support50
are explicitly distinct.

| Arm | Fixed inputs | Learner/objective |
|---|---|---|
| U | Q9 + R319 history support | within-episode pair logistic |
| V | M28 + same history support | identical pair logistic |
| W | M28 + support + R319 seven lag30s differences | primary pair logistic |
| Z | same exact W features/transform | matched state-classification control |
| S/T | saved R319 classification probabilities | unchanged rank references |
| U0/V0/W0 | each pair head's own first score | constant episode timing control |

All new learners: LogisticRegression(C=.1,l1_ratio=0.,solver='lbfgs',
fit_intercept=False,max_iter=2000,tol=1e-8,random_state=20265405). No seed,
regularization, feature, lag/window or objective search. Pair design vectors are
transformed positive state minus transformed negative state and its negative.
At inference score a SINGLE current state with the coefficients; no future
episode mean, future endpoint or evaluation label is needed.

Z uses the SAME50 mixed episodes, exact W transform, identical learner parameters
and total50 weight. Its positive and negative state weights equal the marginal
endpoint mass from the pair manifest, giving half each episode's mass to each
class. W-Z separates pair-relative loss from state-classification loss with
supervised support, weight mass, class balance, transform and intercept held
fixed. Pairwise/state losses use different design vectors by definition; this
is not an isolated causal market treatment. W-T also changes supervised support,
weight/intercept contract and must not be called an objective-only effect.

Report raw relative decision_function scores. They are not absolute success
probabilities: no sigmoid-derived Brier or probability calibration. Copy saved
R319 S/T probabilities solely as monotone rank references. Replay their rank
metrics within1e-9; preserve every saved input, label, censor flag and score.

Chronological diagnostic: May6/7/8 fits only preceding original trajectories,
mixed pairs and matching saved R319 fold transforms. No held/later-day label or
transform use. A no-pair preceding fold yields missing pair scores and an explicit
unscored-fold ledger; never invent a prior score or drop the held observations.

## Fixed metrics and checks

Primary within-episode AUROC on all complete observations, conditional on the78
evaluation episodes with both label classes. Weights match R319: original equal
day/equal episode/equal complete-state weights, conditioned on evaluable episodes.
Report denominator and all655 complete episodes; retaining828 in the ledger does
not mean828 episodes provide timing evidence. Constant-first scores equal.5.
Secondary pooled AUROC/AP and SAME-day BETWEEN-episode rank excluding same-
original-episode pairs. No rank comparison on chosen future-best observations.

Bootstrap whole day and ticker-day multiplicities on ORIGINAL weights,1,000
draws/seeds20265450/20265451, retaining episode identity. Report95% intervals and
valid/invalid draws for W-V,W-U,W-T,W-Z,V-S,W-W0,V-V0,U-U0 across all four rank
metrics. Fixed-fit intervals omit fitting uncertainty. Fixed descriptive phases:
HOT elapsed(0,30],(30,120],(120,300] and available/unavailable lag; no phase selection.

Thirteen checks: >=30 mixed training episodes; >=50 mixed evaluation episodes;
>=4 evaluable dates; W within-episode rank beats V/U/T/Z and.5; W-V improves on a
majority of evaluable dates; separately positive DAY and TICKER-DAY lower timing
CI for W-V,W-T,W-Z,W-W0. Keep every failed criterion. Support/coverage remains
diagnostic and unchanged; no gate relaxation or fallback arm selection. Even
all-pass supports a reused-development ranking hypothesis only, not executable
entry/hold/exit returns or permission to open sealed data.

Persist all pairs/ledger, scored/chronological checkpoints, complete parameters,
coefficient/weight audits, frozen transform replay, input hashes and gates.
Synthetic tests cover censoring, all-pair membership, boundaries, pair/endpoint
weight mass, frozen transforms, current-state-only inference, missing no-pair
folds, raw scores/rank equivalence and whole-cluster bootstraps. Full tests and
critical Ruff; one isolated official cached run after local validation. Reproduce
all JSON values, identities/labels/missingness and raw-score pair ordering within
1e-9, then retire its one-shot workflow. Main/research-request.json stays intact.
