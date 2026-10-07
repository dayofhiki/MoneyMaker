# R327 proposal: observed feature-path innovation before more rate complexity

Not executed or formally preregistered. Register the final input hashes/contract
remotely before any real R327 fit. R326's repaired exact-tolerance
reproduction now passes with error0 across6689 JSON numbers and all seven tables.
Keep that reproducibility prerequisite for the exact selected source/cache. Never relax R326's1e-9 numeric/rank/missingness checks after results.

## Decision and falsifiable hypothesis

R326 improves the matched adjacent objective but F−S timing intervals include0.
Rich F retention median .01957 suggests a fast-rate approximation to the current
rich classifier. Existing T38 already includes one lag30s delta: this does NOT
refute all trajectories. Test whether an **online summary of the observed feature
path** adds conditional information beyond T38/current snapshots and an episode's
fixed initial anchor. Do not force slower rates merely to create memory, search
static interactions, add labels to recurrent state, or interpret rates as real
regimes.

## Frozen support and state update

Retain R326's7806 observed May6–8 training states/7327 complete targets/256 mass,
7066 original adjacent transitions/218 transform episodes, and unchanged19530
8-day evaluation observations/828 identities/all8623 local pairs. No new data,
sealed dates, costs/stops/labels/windows or outcome-dependent clocks.

Use exactly the seven existing LAG_INPUTS:momentum_return_30s,efficiency_30s,
volume_rate_ratio_30s,transactions_rate_ratio_30s,signed_volume_proxy_30s,
reclaim_prior_high_30s, and momentum_acceleration_10_30 (first six with momentum_
prefix). Fix EMA e-folding τ=30s from the already fixed local/lag30s contrast,
not by R326 performance or a τ sweep. For each feature, preserve its own last
finite observation time and EMA. At each ORIGINAL observed clock:

1. If current x and a prior finite EMA exist, emit δ=x−EMA_before_current.
   Also emit matched anchor δ0=x−first_finite_prior_observation.
2. Update EMA ONLY AFTER emission:EMA_new=z*EMA_old+(1−z)*x,
   z=exp(−(current_t−last_finite_feature_t)/1000/30). If no previous finite value,
   emit missing, then initialize EMA and anchor with current x.
3. A missing current feature emits missing δ/δ0 and carries feature state/time.
   Never invent a value from a future observation. Other finite coordinates update.
4. Censored outcomes do not affect any feature state or clock. Latent probability
   still advances at EVERY observed clock using the R326 actual observed gap.

EMA and anchor are reset only at original episode boundaries. First feature
innovations are missing; first PROBABILITY remains pinned canonical G. Raw current
missingness and the innovation missing flags are retained. Do not drop partially
observed trajectories or renormalize their complete-state loss weights.

Training-only support audit: at least one innovation exists at7540 observations,
7072 complete targets/218 episodes, mass190.150914; all seven exist at6990
observations/6546 complete targets/183 episodes, mass130.131680. Individual features
have183–218 complete-target episodes across all3 dates. All7806 observations remain;
missing flags handle limited feature availability. Audit reads no evaluation data
and fits no real model. The audit in prepare_request327_support.py uses elapsed
since each feature's last finite clock and records path-array SHA256 fingerprints.

## Matched models

| Arm | Inputs and probability path | Purpose |
|---|---|---|
| F | T38 + seven EMA innovations; R326 sequence rate model | evolving observed-path hypothesis |
| N | T38 + seven fixed-anchor innovations; same sequence model | matched episode-anchor control |
| S | same T38 + EMA innovation block as F; stationary logistic | history features without predictive-probability carry |
| C | pinned R326 rich sequence F paths/audits | exact current-feature recursive control |
| D | pinned R326 rich stationary S paths/audits | exact same-support snapshot control |
| B/B0 | pinned current/own-first frozen G | absolute/initial controls |

Freeze each current T38 preprocessing block from the corresponding R326 full/past
fold fit. Fit separate7-feature innovation/anchor transforms on the SAME7066
original current-transition states and existing independent mass218. Use existing
weighted1/99% clip,median,mean,scale and missing flags. F/S share the exact EMA
block; F/N share the exact current block and both have90 processed coordinates
(76 current+14 innovation),182 total rate parameters including intercepts.
Separate block transforms prevent adding history from changing current clipping.
The anchor block controls the added feature count, though information and scale
distributions differ; no sole-cause order-only attribution.

F/N retain R326 sequence NLL over original complete weights/mass256, first loss
constant, slope penaltyC=.1, log-domain own-prediction recurrence, actual-time
kernel, deterministic switch-count/exposure initialization and unchanged optimizer/
finite-gradient success rules. S usesC=.2 and original LATER target weights/mass
190.150914, retaining first loss constant, as the matched instantaneous two-rate
penalty. No feature selection, rate bounds, alternate starts or objective search.
Unsupported boundary/single-class fits remain unsupported, not a prior fallback.

Every initial/current G probability reuses pinned canonical R325/R326 bits after
strictly preceding frozen-audit reconstruction verification within1e-9. C/D paths
must replay their pinned official scores exactly within the same tolerance and
rank/missingness scope. No fitting on evaluation or on May5 meta data.

## Chronology, metrics and interpretation

May7 fits F/N/S ONLY on out-of-time May6; May8 ONLY May6–7. May6 has no preceding
meta day: retain all4602 observations and common first,4502 missing later scores
for F/N/S/C/D. Report full7806 rows, common finite scope and fitted May7/8-only
separately. Reuse pinned C/D fold paths rather than fit extra controls after results.

Keep admission as same-day between-original-episode AUROC; also pooled AUROC/AP,
Brier/prior, whole within-episode AUROC, and local30s rank with ORIGINAL eligible
weights, all8 dates, first equality, <=30s/>30s gaps and rate retention quantiles.
Add predeclared complete innovation availability/missingness slices and weighted
feature-history age summaries, descriptive only. No gain may be inferred solely
from dense seconds or first-only chronology.

Bootstrap1000 original whole-day/ticker-day multiplicities,seeds20266150/20266151.
Contrasts F−C,F−N,F−S,F−D,F−B,F−B0,N−C,S−D across all six R326 metrics. Fixed-fit
intervals omit training/initial/EMA-parameter uncertainty (τ is fixed).

36 checks:the same7 support roles;4 aggregate F admission/whole/local/Brier roles
against N/C/S/D/B/(B0 or .5/prior as in R326);majority-date F−B admission;both-scheme
positive F−B admission/whole/local,F−N admission,F−C admission and negative-upper
F−B Brier (12);both-scheme positive F−C whole/local,F−N whole/local,F−S whole/local
(12). Preserve every failure. This is a development gate, not policy promotion.

Interpretations fixed BEFORE results:

- F−C and F−N positive timing intervals support extra evolving feature-history
  value beyond current snapshots and anchoring, conditional on this contract.
- S−D gains with F−S uncertain support a history-feature representation, without
  establishing extra benefit from carrying prior predictive probability.
- F/N/C similarity or unstable gains leave the trajectory addition unsupported;
  do not tune τ, expand interactions, select S as a fallback, or force retention.
- Admission/calibration/chronological failures block overall model improvement
  claims even if some local-timing contrasts are positive.

## Implementation/reproduction requirements

Before real fitting:test feature update-before-emission, last-finite elapsed time,
first-available/missing/censored behavior, constant-feature identity, episode reset,
future/prefix invariance, exact shared current/EMA transforms, original loss/pair
weights, registered optimizer gradients and unsupported May6. Full tests and
critical Ruff; preregister all official319/321/325/326 input digests and source.

Persist original seven tables plus audited feature histories/innovation/anchor/
last-finite clocks, targets/weights/log paths/rates/retention, full/fold fits and
initial replay audits. One successful cached official experiment, authenticated
source/digest, numeric agreement<=1e-9 and exact identities/missingness/tie-aware
ranks, with any failed attempt preserved. Retire the one-shot workflow after
closeout. No raw acquisition, policy change, sealed-date opening or live trades.
