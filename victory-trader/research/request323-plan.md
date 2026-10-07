# R323: joint absolute state and within-episode timing loss

Register remotely BEFORE real-data R323 fits/evaluation. Parent R322 closeout
d3c5b82d7d096e14171b8f6484c90a68a3a43b97/Draft PR95. Follow its recorded
joint-objective proposal. R321 supports nearby price/activity timing beyond an
elapsed control; R322's separate meta-classifier did not improve admission.
Standing user authorization permits this isolated cached research experiment.

## Inputs and shared representation

Nine exact official R319/R320/R321 files are SHA256 pinned in request323-inputs.json.
Use all9325 training observations/8792 complete states/336 complete episodes,
62 positive training episodes and50 mixed episodes/all47825 pairs. Confirm these
counts before fitting; no new cohorts, clocks, labels, costs/stops or acquisition.
Keep828 evaluation identities,19530 observations/18793 complete states/655
complete episodes,78 whole-episode timing and70 local-pair episodes/all8623 local
pairs. Known May development reuse is not independent validation. No sealed June
HOLD/July-August, promotion or live trading.

Use the EXACT R319 T38 causal features (already includes Q9) for rich heads and
G12=Q9+history for controls. No feature union beyond T/G, interaction, new history,
lag/window, seed or regularization search. Copy their exact official R319 audited
per-feature transforms (all336 complete episodes) and verify training-only replay
within1e-9. All labels and every saved score/missing value remain unchanged.

## Fixed convex objective

Let state weights w have sum N=336 under original equal day/episode/state mass;
inherited pair weights v sum M=50 under equal mixed day/episode/pair mass.
x are already transformed individual current states; d=x_positive-x_negative.
For coefficients beta and unpenalized absolute intercept b, minimize:

L=(1-alpha)*sum(w*logistic_loss(y,x*beta+b))/N
  +alpha*sum(v*log(1+exp(-d*beta)))/M
  +||beta||^2/(2*C*N).

Primary alpha=.5; C=.1. This is a fixed1:1 mixture with equal effective total mass
N, equivalently state component mass168 and rescaled pair component mass168. A
positive-first pair plus its reverse with half mass gives the same relative loss;
implement positive-first once and test this identity. Neither47825 pairs nor336
rescaled pair mass changes the independent mixed-episode support50. Intercept
cancels from pair differences; only absolute labels identify b.

Use scipy.optimize.minimize(method=L-BFGS-B,jac=True), deterministic zero start,
maxiter2000,maxls50,gtol1e-8,ftol64*float64 epsilon. Accept only successful finite
optimization with normalized gradient infinity norm<=1e-6. No restart/tolerance
search on market outcomes. Synthetic analytic-gradient finite differences and
alpha=0 equivalence to sklearn LogisticRegression(C=.1,l1_ratio=0,fit_intercept=True,
solver=lbfgs,max_iter2000,tol1e-8) must pass before real fits. Compare real state-only
boundary probabilities with saved R319 counterparts within1e-7, audit errors.

| Arm | Representation | alpha | Role |
|---|---|---:|---|
| J | frozen T38 | .5 | primary joint candidate |
| C | same exact T38 | 0 | matched state-only objective |
| H | frozen G12 | .5 | matched context/history joint control |
| D | same exact G12 | 0 | matched context/history state-only control |

Same all-state transform, training scope, absolute target/prior, regularization
scale and intercept contract. J-C changes the fixed objective mixture including
state component mass, not just an extra fitted feature; report this precisely.
Sigmoid of joint logit is a candidate absolute output evaluated for Brier; it is
not accepted as calibrated by construction. Persist raw logits too. Frozen W raw
scores remain the pair-only timing reference, never sigmoid-calibrated. Own first
J0/C0/H0/D0 scores are broadcast to same later labels as temporal controls.

## Metrics and21 fixed checks

Primary admission metric: SAME-day BETWEEN-episode AUROC on all complete states,
excluding same-original-episode pairs. Original equal day/episode/state weights.
Report pooled AUROC/AP, Brier versus own prior and conditional whole-episode timing
AUROC. Local diagnostic uses the SAME official R32130s pairs and original eligible
day/episode/pair weights; compare RAW logits for tie-aware local pair ranks.
No future-best state selection, new delay, threshold, policy or realized returns.
Descriptive per-day admission/timing/local metrics and saved W/U/T/G ranks remain.

Bootstrap1000 whole day/ticker-day multiplicities, seeds20265750/20265751, preserving
original weights/identity; fixed-fit intervals omit fitting uncertainty. Compare
J-C,J-H,J-D,J-J0,C-D and H-D across admission AUROC, pooled AUROC/AP, Brier,
whole timing AUROC and local pair rank. No individual pair/second resampling.

Twenty-one gates: >=200 complete training episodes; >=30 positive training episodes;
>=30 mixed training episodes; >=50 mixed evaluation episodes; >=30 local evaluable
episodes; >=4 positive evaluation dates (six support checks). J admission rank>C/H/D;
J timing>C/H/.5; J local>C/H/.5; J Brier<C/H/D and own prior (four checks). J-C
admission improves on a majority of evaluation dates (one check). Separate positive
DAY and TICKER-DAY lower admission AND timing CI for J-C (four); positive local CI
for J-C in both schemes (two); positive admission CI for J-H in both schemes (two);
negative upper Brier CI for J-C in both schemes (two). Keep all failures; do not
choose a secondary arm or mixture weight as fallback. Even all-pass is development
evidence only. Any weaker timing than W is reported explicitly, not concealed by
comparisons against state-only C. No model/policy promotion.

Persist every fit/objective/gradient/weight/transform audit, state/first checkpoints,
unchanged828-identity ledger and full pair/local manifests. Synthetic tests cover
solver boundaries, intercept cancellation, orientation/mass, missing/censored
states, causal single-state inference and whole-cluster diagnostics. Full suite
and critical Ruff; one official cached reproduction after local validation;
compare JSON, checkpoint values, missingness and all score-pair ordering within
1e-9 (output file byte hashes must each match their own files). Retire its one-shot
workflow after verification. Keep main/research-request.json unchanged.
