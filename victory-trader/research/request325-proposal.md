# R325 proposal: elapsed-time-consistent latent state evolution

R324's primary recursion fails15/24 gates, especially whole timing and Brier.
Retain state evolution as the next research direction, but test a distinct
mechanism: transition probabilities constrained by actual elapsed seconds.
This is a proposal, not a completed experiment or remote preregistration.
Register the exact optimization/inputs/gates before any R325 model fit/evaluation.

## Hypothesis and training feasibility

R324 treats each observed update as a learned transition with gap as a feature.
It does not enforce a clock-consistent transition law. A continuous-time
two-state approximation may preserve an initial belief across short intervals
while allowing larger changes across long intervals. The target remains the
latent hindsight reachable BASE net>=5 label. Neither a true Markov process nor
Bayesian optimality is assumed or established by R324.

Training-only support audit uses the unchanged SHA256-pinned R319 training
checkpoint, without a new model or held-date tuning.8445 adjacent complete
transitions/286 episodes have gaps1–240s, median2s,90th percentile18s.
Conditional previous-zero support279 episodes; previous-one54. Onset38 and
decay48 episodes across4 dates.>30s transitions have only10 onset and9 decay
rows: report that limitation rather than use separate gap-bin models.

## Fixed candidate contract to preregister

Learn nonnegative onset and decay rates a(x),b(x), with log-linear rates in
current causal features and a training-fitted intercept. For actual gap d seconds,
let s=a+b, z=exp(-s*d), and pi=a/s. Use

q01=pi*(1-z), q11=pi+(1-pi)*z,
p_current=pi+(p_previous-pi)*z.

The rates are an approximation conditional on features observed at the current
completed clock. They do not reconstruct unobserved intervening states or promise
correct causal dynamics during a gap. With features fixed, two updates of lengths
d1,d2 must equal one update of length d1+d2. Require q11>=q01, probability bounds,
zero-gap identity and large-gap stationary limit in synthetic tests. This only
holds for constant inputs; arbitrary inserted market observations can change inputs.

Use the same four recursive roles: primary T38 rates, G12 context/history rates,
three HISTORY age rates, and constant rates. Actual d enters analytically, not
as a selected interval or gap covariate. Freeze current G and own-first G controls,
plus R324's fixed rich recursion as a descriptive predecessor reference. All
models share the original G initial prediction and inherited clocks/labels.
No posterior reset, held labels, censor flags or outcome-derived state at inference.

Jointly fit the two rates by adjacent-transition Bernoulli likelihood, using
the inherited separately recomputed previous-zero/one episode weights (279/54
mass), with C=.1 L2 on slopes and unpenalized rate intercepts. Fit preprocessing
on all8445 current transition states/286 episodes, copied to both rate heads.
Use a30s rate unit only for numerical representation, with a free intercept;
do not choose a rate timescale based on held performance. Fix optimizer, analytic
gradient, tolerances, stable probability/log-likelihood arithmetic and deterministic
training-only initialization in the preregistration. Convergence failure is a
reported failure, not an optimizer/grid search. Do not rebalance rare switches.

For May6/7/8 diagnostics, fit rates and transforms strictly on earlier original
training days and use matching G initialization. Keep every held observation and
explicit no-support states. Single-class/boundary behavior must be registered
before execution rather than inventing a rate from held information.

## Evaluation and decision

Same828 evaluation identities/19530 observations/18793 complete states; same
78 mixed timing episodes and70 local episodes/all8623 original30s pairs. Retain
all original censoring and label support. Admission remains same-day
between-episode AUROC; also report whole/local timing, AP, Brier/common prior,
all dates, common first predictions and fixed >30s diagnostic.

Retain R324's24 check roles and both1000-draw whole day/ticker-day interval
schemes on original weights, with preregistered new seeds. A gain over failed
R324 alone is insufficient: primary rates must improve the frozen current G
baseline and matched context/age/constant controls without a Brier loss. Keep
W's .671630 whole/.599223 local rank as a descriptive frozen benchmark.
Intervals omit fitting uncertainty; repeated May use is not fresh validation.

Run meaningful causal/numerical tests and the full suite before real fits;
persist all rates, transforms, q heads, recursive paths, weights and ledgers.
Require one cached official reproduction, exact identities/missingness/ranks and
numeric tolerance1e-9, artifact/source authentication, then retire its workflow.
No raw acquisition, threshold/window/feature search, fallback arm selection,
main policy change, June HOLD/July-August access, promotion or live trades.
