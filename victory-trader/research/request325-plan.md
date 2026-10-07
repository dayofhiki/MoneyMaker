# R325 preregistration: elapsed-time-consistent latent opportunity rates

Register remotely BEFORE any real R325 fit/evaluation. Parent R324207a51330b5dce98
ba2e7ae1257a198e5355cf91/Draft PR97, official cached run37602021567. User authorizes
execution, analysis and next design under operating-authorization.md. R324 fails
15/24 gates; this changes the model contract, not its gates or initial prediction.

## Fixed population, provenance and independence

Twelve exact files from official R319/R321/R324 are SHA256 pinned in
request325-inputs.json. Training9325 observed states/347 observed episodes,
8792 complete states/336 complete episodes. Original adjacent complete transition
manifest is identical to R324:8445 transitions/286 episodes;0→0=7553,0→1=88,
1→0=99,1→1=705. Onset38 and decay48 independent episodes across all4 training
dates. Previous-zero supervision7641 rows/279 episodes and previous-one804/54.
Training-only gap audit: median2s,90th percentile18s,max240s. Do not create
transitions by skipping an incomplete original endpoint or assume regular clocks.

Same828 evaluation identities,19530 observations,18793 complete states/655
complete episodes,78 mixed timing and70 local episodes/all8623 inherited30s
pairs. Preserve every label, censor flag, feature, missingness and old score.
All observations, including censored outcomes, enter inference. No acquisition,
label/cost/stop changes, sealed June HOLD/July-August, promotion or live trades.
May is reused development; hindsight reachable BASE net>=5 is a latent research
target, not an observed regime or realized return.

## Rate models and prediction-only evolution

For current causal x, log rates per30s unit are eta_a=x beta_a+i_a and
eta_b=x beta_b+i_b. Rates per second a=exp(eta_a)/30,b=exp(eta_b)/30. For actual
elapsed d=(current_t-previous_observed_t)/1000, s=a+b,z=exp(-s*d),pi=a/s:
q01=pi*(1-z),q11=pi+(1-pi)*z;
p_current=pi*(1-z)+p_previous*z. Every p_previous is a previous PREDICTION.

| Arm | Current rate features | Role |
|---|---|---|
| F | inherited T38 | primary rich rate evolution |
| H | inherited G12 | matched context/history rates |
| A | three inherited HISTORY features | matched age/history rates |
| K | intercepts only | learned constant rates |
| B | exact frozen R319 current G classifier | absolute baseline |
| B0 | exact frozen G first broadcast | initial-only control |
| P | official R324 rich recursive F, unchanged | descriptive predecessor |

Actual d enters analytically; no gap feature, window or interval search. Fit
both rates jointly. Changing parameterization, joint likelihood and removing
the gap covariate prevents attributing results solely to clock consistency.
Every recursive arm starts from the SAME frozen R319 G first p. Replay full G
and chronological-fold G saved probabilities within1e-9. All first predictions
are unchanged; do not claim first-clock improvement. P is replayed from pinned
official checkpoints, never refitted or chosen as fallback.

Rates conditional on currently completed features are an approximation during
the preceding gap. They do not reconstruct unseen states or establish a Markov
process/optimal Bayesian filter. Constant x must obey the composition law
update(d1) then update(d2)=update(d1+d2), zero-gap identity, large-gap stationary
limit,0<=q01<=q11<=1. This is only for fixed inputs; extra observations can change x.
No true past/current/future label, outcome mask, held-date fit, future mean,
reset, threshold or posterior clamping enters inference. Causal gaps use all
original observations. First gap/q heads are missing, not a fabricated interval.

## Frozen preprocessing, likelihood and optimizer

For each representation fit existing weighted1/99% clip/median/mean/scale and
missing flags on all8445 transition-current states with equal day/episode/state
transform weights of total286. Both rate heads copy that transform. K has zero
slopes and two free intercepts. Supervision uses unchanged R324 conditional
weights separately recomputed within previous-zero/one support, totals279/54.
The joint objective is
sum(w_i * negative log P(y_current_i|y_previous_i,x_i,d_i))
+ (||beta_a||^2+||beta_b||^2)/(2*C), C=.1; intercepts unpenalized.
Divide the complete objective and analytic gradient by total supervision mass
for numerical optimization without changing the minimizer or C. No balancing,
oversampling, treating transitions as independent episodes or weight retuning.

Initialize slopes0 and each rate from OWN training conditional weighted switch
count / weighted elapsed exposure, represented per30s unit. This is an
initialization only, not a claimed rate estimate or fit. SciPy L-BFGS-B,
maxiter2000,maxls50,ftol1e-13,gtol1e-8, no bounds, one deterministic start.
Require success, finite objective/parameters and normalized gradient infinity
norm<=1e-6. No retry, alternate optimizer or hyperparameter search. Use stable
log-sum-exp, log(-expm1(-h)), small-h series, exact large-h limits and analytic
log-likelihood gradients. No probability clipping in likelihood or inference.
Nonfinite persisted rate/prediction or failed full optimization is a recorded
experiment failure; save audit and do not evaluate a replacement model.

Both previous conditions and all four transition cells must have training
support for a finite two-rate fit. Empty/single-class/boundary support reports
unsupported (no fabricated finite rate); keep held rows and common initial,
leave later recursive outputs missing. A chronological diagnostic uses common
finite core-rate support explicitly if needed. Full evaluation requires all
core predictions finite on every original observation, otherwise fails.
May6/7/8 rates/transform fit ONLY preceding original training days with matching
frozen G initial model. Keep all7806 chronological held observations.

## Metrics,24 gates and reproduction

Admission is same-day BETWEEN-original-episode AUROC on all complete evaluation
states, excluding within-episode comparisons. Also report pooled AUROC/AP,
Brier/common frozen G prior, conditional whole timing AUROC, inherited local30s
pair rank with ORIGINAL eligible day/episode/pair weights. Report all8 dates,
common first and fixed gap<=30s/>30s diagnostics. No entry policy/profit claim.

Bootstrap1000 whole day/ticker-day multiplicities on original weights, seeds
20265950/20265951, fixed fits. Compare F-H,F-A,F-K,F-B,F-B0,H-A and descriptive
F-P on all six metrics. Do not treat pairs/seconds independently. Intervals
omit rate/initial-model fit uncertainty.

Retain R324's24 roles unchanged: seven support checks (onset>=20,decay>=20,both
>=3 training dates,transition episodes>=200,mixed eval>=50,local eval>=30,
positive eval days>=4); F admission>H/A/K/B/B0; F whole and local>H/A/K/B/.5;
F Brier<H/A/K/B/B0/prior; F-B admission improves on a majority of dates; both
scheme positive F-B admission/whole/local intervals, positive F-H admission,
positive F-A admission, negative upper F-B Brier (twelve interval checks).
Passing is development evidence only. Gain over failed P alone is insufficient.
Keep every failure and frozen W .671630 whole/.599223 local as descriptive rank.

Before real fits, meaningful synthetic analytic-gradient/limits/composition,
conditional weight/transform, label/censor/future mutation, causal-only inference,
unsupported scope and chronology tests plus full suite/critical Ruff must pass.
Persist original transition manifest/ledger, training/all eval/fold/first states,
828 ledger, both rates/log rates/stationary probabilities/q heads/paths, fit
weights/transforms/objectives/gradients and source/input audits. One cached
official reproduction after local validation; compare JSON and all seven parquet
values/missingness/identities and tie-aware probability ordering within1e-9.
Authenticate artifact ZIP/source; retire one-shot workflow; record findings and
next proposal without changing the tested contract after seeing results.
