# R326 preregistration: prediction-only sequence likelihood

Register remotely BEFORE any real R326 fit/evaluation. Parent R32597bc53d260cf
24ea2f9afb3f5722deeae87e4a45/Draft PR98; cached official run37618872764. Standing
research authorization applies. R325 fails15/24 checks; test a learning-objective
hypothesis while retaining the actual-time rate architecture and matched controls.
Conditional filtering can be valid under correct specification; no diagnosed
teacher-forcing bug or guaranteed trajectory benefit is assumed.

## Frozen data and initial predictions

Ten official R319/R321/R325 files are SHA256 pinned in request326-inputs.json.
NEW training is ONLY the official R321 chronological May6/7/8 states:7806 observed
states/266 episodes,7327 complete states/256 episodes,51 positive episodes.
Adjacent complete support7066 transitions/218 episodes:0→0=6265,0→1=77,1→0=85,
1→1=639. Onset33 and decay41 episodes, all3 dates; previous-zero conditional
support6342 rows/214 episodes, previous-one724/45. Counts are not independent
seconds/pairs. Do not use May5 as new meta training or skip incomplete endpoints.

Reconstruct each training day's G initial/current scores from its strictly
preceding R319 fold audit and replay saved G within1e-9. Use each episode's own
first probability, never a true label or G in-sample full-training score. Held
8-day evaluation uses unchanged full May5–8 G, as R324/R325; fold-initial model
differences are a limitation. Every recursive/static control has identical first p.

Same828 evaluation identities/19530 observed states/18793 complete states/655
complete episodes,78 mixed timing and70 local episodes/all8623 original30s pairs.
Retain every old clock, feature, score, label, censor flag and missingness. Process
every observation in prediction history, including censored outcomes. No raw
acquisition, window/threshold/feature search, costs/stops/labels/main policy change,
sealed June HOLD/July-August, promotion or live trades. Reused May development and
latent hindsight reachable BASE net>=5 targets, not fresh validation/realized profit.

## Models, preprocessing and objective

| Arm | Features/learning | Role |
|---|---|---|
| F | T38 rates, prediction-only sequence NLL | primary |
| H | G12 rates, same sequence NLL | context/history control |
| A | three HISTORY rates, same sequence NLL | age/history control |
| K | two constant-rate intercepts, sequence NLL | constant control |
| M | T38 rates, R325 adjacent endpoint likelihood | matched objective control |
| S | T38 current-only stationary logistic | matched memory-free control |
| B/B0 | unchanged current G/own-first G | absolute/initial controls |
| P | unchanged official R325 rich F | descriptive predecessor |

Same R325 rate law: log rates eta_a=x beta_a+i_a,eta_b=x beta_b+i_b per30s unit;
a=exp(eta_a)/30,b=exp(eta_b)/30,h=(a+b)*actual_gap_s,z=exp(-h),pi=a/(a+b),
q01=pi*(1-z),q11=pi*(1-z)+z,p_t=pi*(1-z)+p_previous*z.
No gap feature, recurrent label/reset, posterior clamp or invented time interval.
Constant-input composition, zero-gap identity and stationary limit must still pass.

All feature transforms use the SAME7066 transition-current states/218 episodes
and existing independent equal day/episode/state weighted1/99% clip/median/mean/
scale/missing flags. F/M/S copy the exact rich transform. Both rate heads share
one transform. Primary recurrent history is all7806 observations; complete labels
select LOSS TARGETS only, not prediction steps.

Compute original complete-state weights once, mass256; set censored-target weights
0 while retaining their observations. Objective is sum(w_i*NLL(y_i,p_i)) +
(||beta_a||²+||beta_b||²)/(2*.1), divided by complete mass256 for optimization.
Include first loss and its original weight/mass; its fitted gradient is zero.
Initialize slopes0 and each rate by original conditional weighted switch count /
elapsed exposure, represented per30s. No balancing, rescaling first-state loss,
multistart or alternative initialization. Adjacent M keeps conditional mass214/45,
total259 and C=.1. Objective weighting/supervision also changes with sequence loss,
so this is a controlled contract comparison, not a sole causal attribution.

Use stable log-probability/complement recurrences and analytic reverse gradients
through own predictions, including censored steps. Log-domain stationary/carry
mixture weights propagate adjoints; no teacher-forced training state. First
adjoints have no fitted input. Kernel uses R325 stable small/large-h limits.
SciPy L-BFGS-B,maxiter2000,maxls50,ftol1e-13,gtol1e-8,no bounds,one deterministic
start. Require success, finite parameters/objective, normalized gradient infinity
norm<=1e-6. No retry/search or likelihood probability clipping. A numerical/full-fit
failure is saved and stops replacement evaluation. Both conditional supports/all
four transition cells required for finite rate initialization; boundary/empty
support is explicit unsupported, retaining first and missing later predictions.

S has intercept and processed T38 coefficients, LogisticRegression(C=.2,
l1_ratio=0,lbfgs,max_iter2000,tol1e-8,fit_intercept=True,random_state20266005).
Use original full complete weights for LATER states only, with no renormalization;
first loss remains fixed. Its penalty matches the instantaneous rate limit:
min(||beta_a||²+||beta_b||²)=||beta_a-beta_b||²/2, hence C=.2. Unsupported
single-class/empty later targets and convergence warnings are recorded, not a
fallback prior. Fast-rate/memory collapse is an outcome, not grounds to tune rates.

## Chronology, diagnostics and32 checks

May7 rates/S train only on out-of-time May6 records; May8 only May6–7. May6 has
NO preceding out-of-time meta-training day: retain every row and common first,
report unsupported later scores for F/H/A/K/M/S. Never fabricate a fit or silently
discard a date. All7806 diagnostic rows remain. Report each fold, common finite
scope and fitted-May7/8-only metrics separately. Main8-day result requires finite
core predictions on every original observation. P is replayed from official
R325 states/folds and retains its earlier, larger training support; descriptive only.

Admission remains same-day BETWEEN-original-episode AUROC excluding within pairs;
also pooled AUROC/AP,Brier/common G prior, whole timing and inherited local30s rank
with ORIGINAL eligible day/episode/pair weights. Report all8 dates, first equality,
fixed gap<=30s/>30s and weighted retention-z mean/10/50/90% quantiles on complete
nonfirst states. Do not call an instantaneous/stationary gain a memory benefit.

Bootstrap1000 whole day/ticker-day multiplicities on original weights,seeds
20266050/20266051. Contrasts F-H,F-A,F-K,F-M,F-S,F-B,F-B0,H-A,F-P across all6
metrics. Fixed-fit intervals omit rate/static/initial-model fitting uncertainty.

Keep R325's24 roles, adding M/S to four aggregate admission/timing/local/Brier
comparators. Seven support checks: onset>=20,decay>=20,both>=3 training dates,
transition episodes>=200,mixed eval>=50,local eval>=30,positive eval days>=4.
F admission>H/A/K/M/S/B/B0; F whole/local>H/A/K/M/S/B/.5; F Brier<H/A/K/M/S/B/B0/
common prior. Majority-date F-B admission. Both schemes positive F-B admission/
whole/local,F-H admission,F-A admission and negative upper F-B Brier (twelve).
Add both-scheme positive F-M whole/local (four) and F-S whole/local (four):32 total.
Preserve every failure; no fallback selection. W whole .671630/local .599223
remains descriptive frozen raw-score evidence, not a probability model.

Before real fitting: sequence gradients across multi-step/censored histories,
first-loss constancy, label/censor/future mutations, causal-only inference, static
penalty equivalence, transform/weight independence, unsupported May6, scope and
full suite/critical Ruff. Persist transition manifest/ledger, enriched training/
all eval/fold/first states,828 ledger, weights/initials/log paths/rates/q/retention,
fits/objectives/gradients and input/source provenance. One cached official replay,
JSON/all seven parquet identity/missingness/value/tie-aware probability-rank
agreement within1e-9, authenticated source/ZIP, then retire one-shot workflow.
Passing development checks still cannot authorize promotion or sealed dates.
