# R328 proposal: an observability-gated trajectory residual on frozen G

Register remotely BEFORE any real R328 fit/evaluation. Parent R327 8011bd6671f982834ba4feb264770bbf76d7977f / Draft PR100. Standing operating-authorization.md applies. R327's single official source6c2bb5518187c41d7a59a3e1d9d3085fd9ec50dd/run37652470621 and both authenticated artifacts11497031370/11497126349 reproduce8,941 JSON numbers and all eight tables exactly. This hypothesis uses reused May development, not fresh evidence. All inputs are pinned in request328-inputs.json.

## Why this experiment

R327 F passed 9/36 checks. Its extra EMA history did not establish whole/local
timing gains over current-feature C, fixed-anchor N or same-feature stationary S.
Brier was worse than frozen G under both bootstrap schemes. The predeclared
availability diagnostic shows F local timing .585260 versus C .550842 where all
seven innovations exist, but .473385 versus .637801 in partially observed rows.
These are overlapping episode slices, with just 15 local episodes in the partial
slice and no slice intervals. They do not prove that missingness caused failure.

Test whether evolving observed history adds a useful *conditional correction* to
an unchanged, strictly out-of-time absolute-quality predictor. Keep G fixed to
avoid relearning its admission/calibration while testing the path summary. Gate
the correction by causal feature availability; evaluate all original rows and
pairs. Compare against the same gate with anchor, current-feature and mask-only
corrections. This tests observed state evolution through feature history, without
claiming extra benefit from carrying the last predicted probability. T38/G already
contains history and lag deltas. No static interaction expansion or rate/τ sweep.

## Frozen inputs, state and gate

Retain 7,806 observed May6–8 training states, 7,327 complete targets and original
mass256; 7,066 original transition-current preprocessing rows/mass218; all19,530
evaluation observations, 828 identities, 18,793 complete targets/655 episodes,
all8,623 local pairs, 78 mixed and70 local episodes. Preserve clocks, labels,
censoring, costs/stops, firsts and sealed dates. No raw market requests.

Reuse exactly R327's seven LAG_INPUTS, τ30s, emit-before-update EMA innovations,
first-finite anchors, per-feature last-finite times, missing-state carry and
episode reset. All observed clocks, including censored targets, update finite
features. No new lookback, availability threshold, imputation rule or feature.

Define m_t=1 only if ALL seven EMA innovations are finite at the current original
clock; otherwise0. This requires current values and prior finite feature history.
The EMA/anchor availability masks are identical on the training audit. The gate
reads no targets and no future. Its full evaluation behavior is fixed before R328
outcomes. It is an architecture choice motivated by R327, not a selected winning
evaluation subset: all inactive rows/pairs remain in evaluation and loss.

G baseline g_t is the canonical pinned R326_B current probability: strictly
preceding G folds for May6–8 training, frozen May5–8 G for evaluation. Reconstruct
the frozen audit within1e-9 then reuse saved bits, preserving ranks. Never use
full-fit R326 C or R327 F training predictions as an offset: those meta heads saw
the training targets. They are not out-of-time training offsets. First g_t and
all original G priors are unchanged. No G refit or global intercept calibration.

## Four residual arms and exact controls

For each fitted residual head, use

    p_t = sigmoid(logit(g_t) + m_t * u_t · β).

Every gate0 prediction is a direct bit-exact copy of g_t, rather than a logit
round trip. Gate1 zero correction also copies g_t exactly. All firsts have gate0.

| Arm | u_t at gate1 | Free parameters | Purpose |
|---|---|---:|---|
| F | seven standardized EMA innovations | 7 | evolving observed path |
| N | seven standardized first-anchor innovations | 7 | fixed episode anchor |
| S | seven standardized current LAG_INPUTS | 7 | current-coordinate correction |
| M | scalar1 | 1 | eligibility-only calibration correction |
| B | pinned current G, no residual | 0 | absolute-quality control |
| B0 | pinned own-first G, no residual | 0 | initial-only control |

F/N/S have no intercept and no missing-indicator coefficients; M is a penalized
gate-specific intercept only. All four use the EXACT same gate and loss/offset
support. M has fewer parameters, so F−M is a diagnostic of feature information
beyond the gate, not a matched-complexity causal attribution. F−N/F−S compare
equal coefficient counts with different information and separate raw transforms.

For each of F/N/S, fit the existing weighted1/99% clip, median, mean and scale on
ALL original transition-current rows and their original transform weights. Use
only the first seven standardized numeric columns of the transform; discard its
seven missing flags. Do not filter these transform rows by the gate. Gate0 design
rows are EXACT zeros, never NaN*0; gate1 inputs must be finite. Freeze transforms
from preceding training dates for chronological folds. No current-rich block is
refitted: its influence remains in fixed G.

## Objective, optimizer and unsupported cases

Fit all original complete targets, including firsts/inactive loss constants:

    L(β) = [Σ_t w_t {logaddexp(0, η_t) − y_t η_t}
            + ||β||² / (2C)] / Σ_t w_t,
    η_t = logit(g_t) + X_t β; X_t = 0 when gate0; C=.1.

Use original complete-target weights/mass256, not mass130.131680 of the active
subset. Compute constants too. Gate0 loss contributes no gradient. The M scalar
is penalized identically. C=.1 follows the existing one-vector G/residual slope
contract; R327 stationary S's C=.2 matched a two-rate penalty, which is absent here.

Use stable logits/NLL and an analytic gradient. One zero-coefficient start;
SciPy L-BFGS-B maxiter2000,maxls50,ftol1e-13,gtol1e-8, no bounds. Require solver
success, finite coefficients/objective and normalized gradient infinity<=1e-6.
No alternate starts, regularization/τ/gate search, fit selection or retry. Require
all used G scores strictly interior (0,1), not clamped; otherwise stop and audit.
Empty/one-class active training is unsupported, not a prior/model fallback.
Any genuine numerical full/fold failure saves audit and stops evaluation.

## Training-only feasibility and chronology

prepare_request328_support.py reads only two pinned training files and fits no
real model. All-seven gate:6,990 observations,6,546 complete targets/183 episodes,
41 positive episodes and130.1316797413857 original weight mass; both classes over
all3 dates. Inactive:816 observations,781 complete targets,125.8683202586143 mass.
There are6,540 active transform-current rows/mass141.1485585385626 within the
unchanged7,066/mass218. Prefix G active range [.004091966,.264023424] is interior.
Episode counts across masks/classes overlap; their row/weight partitions add.

May7 fits ONLY May6 (69 active target episodes,15 positive; original complete
mass95,active52.7177466). May8 fits ONLY May6–7 (129 active episodes,34 positive;
complete mass179,active94.2330486). Baseline G for every training state remains
its original strict-past probability, not refitted to the residual training dates.

May6 has no preceding meta day: all4,602 rows remain. With no residual fit, return
G exactly for its284 gate0 rows, but mark its4,318 gate1 residual-head scores
unsupported/NaN. This is the defined zero-design branch, not a fabricated zero
fit at gate1. B/B0 stay finite. Report full7,806 rows,3,488 common finite rows,
per-date availability and fitted May7/8-only separately. Do not compare the full
chronological aggregate as if all dates had learned heads, or drop May6.

## Metrics, intervals and development checks

Same original-weight admission AUROC, pooled AUROC/AP, whole-episode timing,
local30s rank, Brier/prior, per-date counts and first equality. Evaluate all original
rows/pairs; report gate availability slices and <=30s/>30s gaps as descriptive
diagnostics only. Never claim improvement from the active subset alone.

Bootstrap1,000 original whole-day and ticker-day multiplicities, seeds20266250
and20266251. Fixed fitted models, original independent weights. Contrasts F−N,
F−S,F−M,F−B,F−B0,N−B,S−B,M−B over all six inherited metrics. Intervals omit fitting
and gate-design uncertainty and reuse development data.

Freeze40 checks before fitting:

- Nine support/integrity roles: >=150 active training episodes; >=30 active
  positive episodes; both active classes across>=3 training dates; >=200 original
  transform episodes; >=50 mixed evaluation episodes; >=30 local episodes;
  >=4 positive evaluation dates; exact gate0/first G copies; strict-prefix G replay.
- Four aggregate roles: F admission>N/S/M/B/B0; F whole>N/S/M/B/.5;
  F local>N/S/M/B/.5; F Brier<N/S/M/B/B0/prior.
- One majority-date F−B admission role.
- Eight interval roles: both schemes F−B admission/whole/local positive and
  F−B Brier upper bound<0.
- Twelve interval roles: both schemes positive F−N/F−S/F−M whole/local.
- Six interval roles: both schemes positive F−N/F−S/F−M admission.

All failures remain. A useful evolving-path claim requires timing gains against
anchor/current/mask controls under both schemes, with admission/calibration checks
also passing. N/S/M winning is not a fallback-selection rule. Failure means this
registered conditional path addition remains unsupported; do not tune gates,
select a subset or force memory. Passing development gates still authorizes no
promotion, live trading or sealed-date evaluation.

## Before execution and closeout

Pin authenticated official327/326/321/319 source/artifact hashes and raw inputs.
Reproduce the training path/gate/prefix G fingerprints in the support audit before
fitting. Test label-free history, future/prefix invariance, missing-time carry,
exact zero-design G copies (including unsupported folds), same gate/control
dimensions, no missing coefficients, original weights and finite-difference
offset gradients including constant rows. Full tests and critical Ruff first.

Persist original population/transition/pair ledgers, feature histories, gate,
offsets, design/audits, predictions/logits, original weights and full/fold fits.
One cached official replay; authenticate all artifacts/source; JSON/coefficients/
tables <=1e-9 with exact identities/missingness/tie-aware ranks. Preserve failed
attempts and do not relax tolerance after outcomes. Retire the one-shot workflow
only after verified closeout. No main merge, policy change or promotion.

## Registered persistence/official packaging

Persist the inherited seven original-population tables, feature-history table
(scope/source indices, all63 R327 path quantities, original target weights) and
a ninth design-matrix table (training/evaluation/chronological scopes, gate,
G offset, original weights and seven F/N/S numeric coordinates plus M scalar).
Keep complete full/fold fit audits, corrections/logits and all probability logs.
Numeric1e-9 agreement and EXACT score/correction/logit tie-aware ranks/missingness
apply to all nine tables and all JSON/fit-audit numbers. No relaxed checks.

Before any real fit, fix two-artifact transfer packaging for the same single
official run: primary contains all output except training-states/feature-history;
secondary contains those two large tables plus identical source/environment files.
Authenticate BOTH ZIP digests/source and identical duplicated metadata, then
merge original names and verify all nine tables. This anticipates the32MiB
per-archive transfer limit; it does not change rows/models/weights/tolerances.

Loss gradients ignore censored targets by indexing positive weights before
label arithmetic. Unsupported full fits save audit and stop. Unsupported folds
retain gate0 G and gate1 NaN; genuine numerical failures stop with audit.
Parameter schemas/gradient tolerance are the same for all full and fitted folds.
