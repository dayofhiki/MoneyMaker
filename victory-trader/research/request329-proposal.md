# R329 proposal: accumulate current evidence on frozen G

Not executed or formally preregistered. Register the final source/input digests
and complete contract remotely before any real R329 fit/evaluation. R328's single
official run37657314310/sourceaa031411f1ab790c2c10d5f0a92dc9aa9225aead reproduces
2,755 JSON numeric fields and all nine tables with error0. The initial local
design-table truncation and independent serialization repair are recorded; no
refit or verification relaxation. This remains reused May development.

## Falsifiable hypothesis and current bottleneck

R328 F's EMA *innovation* improves local timing over G and mask-only M but fails
27/40 checks. F is weaker than anchor innovation N on whole timing and current
coordinate S on whole/local timing under both bootstrap schemes. S−G whole/local
intervals are positive, while S admission/Brier improvement is unproved. These
controls are informative, not selected replacements for a failed primary arm.

Extra history is not automatically useful when encoded as current minus a past
average. Test a different state evolution: **accumulating the level of current
observed evidence** instead of measuring deviation from its past. The comparison
is a fixed memory representation versus the same instantaneous evidence and a
fixed first-eligible anchor, on the same G offset/gate/weights. No feature or static
interaction expansion, label-feedback recurrence, forced CTMC retention, τ search
or claim that a market regime has been identified. G/current coordinates already
contain history. This experiment asks about added accumulated evidence, not the
last predicted probability.

## Unchanged population, gate and absolute-quality predictor

Same7,806 observed May6–8 states/7,327 complete targets/mass256,7,066 transition
currents/mass218 and all19,530 evaluation observations/18,793 complete targets,
828 identities/655 complete episodes,78 mixed/70 local episodes/all8,623 pairs.
Costs/stops, labels, censored clocks, firsts and sealed dates remain unchanged.

Keep the EXACT R328 all-seven finite EMA-innovation availability gate. Its R327
raw feature-path update remains unchanged. Gate0 predictions copy current G bits
exactly; all gate0 rows/pairs/loss constants remain, rather than restricting the
evaluation population. Baseline G is the original strict-past canonical training
G and frozen May5–8 evaluation G. Verify its reconstruction within1e-9, then reuse
bits. Never use a full-fit R328 meta prediction as a training offset.

## Fixed observed-evidence state

Freeze the corresponding full/preceding-fold R328 S preprocessing: same seven
current LAG_INPUTS, original transition-current rows/weights, weighted1/99% clip,
median/mean/scale; numeric columns only. Replay this transform within1e-9. Let
x_t be its seven-coordinate current vector when the shared gate is1.

Reset vector state, anchor and last eligible clock at the original episode start.
At each ORIGINAL observed clock:

1. Gate0: carry evidence state/anchor/last eligible time; residual design is exact
   zero and prediction is G. No update using median-filled missing coordinates.
2. First eligible clock: initialize e_t=x_t and a_t=x_t, recording current time.
   The accumulated/current/anchor basis vectors are EXACTLY equal here.
3. Later eligible clock: z=exp(−(t−last_eligible_t)/1000/30); update
   e_t=z*e_before+(1−z)*x_t; carry a_t unchanged and record current time.
4. Predict using the UPDATED e_t, which includes only current/past evidence.
   Censored targets still update whenever their observed features meet the gate.

τ30s is fixed from the existing local/history contrast; no alternate duration or
cap. Each gap uses elapsed since the last eligible vector, including intervening
gate0 clocks. State exists during gate0 but does not change its G-only prediction.
Persist pre/post evidence, first anchor, last eligible time, gate and all bases.
The hypothesis may fail through stale/noisy evidence or delayed burst response.

## Two new heads and pinned controls

| Arm | Gate1 residual basis | Free parameters | Purpose |
|---|---|---:|---|
| F | accumulated evidence e_t | 7 | evolving evidence-level memory |
| N | fixed first-eligible evidence a_t | 7 | matched anchor control |
| C | pinned official R328 S prediction/audit | 0 new;7 existing | current evidence control |
| M | pinned official R328 mask-only prediction/audit | 0 new;1 existing | gate-only correction control |
| B/B0 | pinned current/own-first G | 0 | quality/initial controls |

F/N share the EXACT frozen numeric transform and gate. Fit separate slope vectors
without intercept/missing coefficients; p_t=sigmoid(logit(G_t)+basis_t·β).
Gate0 and exactly zero corrections use copied G bits. F/N/C have equal coefficient
counts, but history representations differ; no sole-cause order-only attribution.
C/M are direct official328 paths in training/evaluation/chronology, not new fits.
Training C/M full-fit predictions are descriptive controls, never new offsets.

Use the R328 original-complete-target offset NLL, whole mass256 including first/
gate0 constants, L2 C=.1 for seven slopes and no intercept. Because the fixed
evidence state is a linear function of the inputs, the score remains linear in
β and the objective is convex. Do not normalize by active/carried rows. Stable
NLL and analytic gradient; one zero start; L-BFGS-B maxiter2000,maxls50,ftol1e-13,
gtol1e-8,no bounds. Finite parameters/objective, success and normalized gradient
infinity<=1e-6 required. Strictly interior G, no clipping. Unsupported empty/
single-class fits remain unsupported. Numerical failure audits and stops; no retry.

## Training feasibility and chronological scope

prepare_request329_support.py reads only pinned training states, official328 fit
audit and source. No evaluation data or real model fitting. Shared gate has6,990
observations/6,546 complete targets/183 episodes,41 positive episodes. After a
previous eligible clock,6,799 observations/6,364 complete targets from162 episodes,
38 positive episodes,carry113.746935476 of the original mass256 over all3 dates.
All these carried observations differ from current-only coordinates. The191 first
eligible observations include182 complete targets and mass16.384744266. Episode
counts overlap and first eligible observations can have censored outcomes.

May7 preceding fit uses only May6:62 carried target episodes/14 positive, original
complete mass95. May8 uses only May6–7:114 carried episodes/31 positive, mass179.
Replay the saved S transform for each exact preceding scope. Save full/fold path
fingerprints before fitting; audit proves exact first-eligible vector equality.

May6 lacks preceding meta fit: all4,602 rows remain,284 gate0 G predictions and
4,318 unsupported gate1 predictions for F/N/C/M. All7,806 chronological rows and
3,488 common finite rows remain. Report fitted May7/8-only separately; no fake
zero-coefficient learned head and no date removal. C/M share this exact scope.

## Metrics, intervals and 40 fixed checks

Same original-weight admission AUROC, pooled AUROC/AP, whole timing, local30s
rank, Brier/prior, per-date support and first equality. All original rows/pairs;
availability, first-eligible/subsequent and gap slices are descriptive only.
Report local pairs with BOTH endpoints eligible as a descriptive diagnostic:
fixed anchor N preserves G's order within each episode on those endpoints. This
helps separate gate-boundary shifts from evolved evidence; do not select this
subset as a validation population or infer a gain from it alone.

Bootstrap1,000 original whole-day and ticker-day multiplicities, seeds20266350
and20266351, fixed fits and original weights. Contrasts F−N,F−C,F−M,F−B,F−B0,
N−C,N−B,C−B,M−B across all six inherited metrics. No fit/design uncertainty is
included; no fresh validation is asserted.

Forty checks:9 support/integrity (>=150 carried training target episodes;>=30
carried positive episodes; both carried classes across>=3 dates;>=200 original
transform episodes;>=50 mixed evaluation episodes;>=30 local episodes;>=4 positive
dates;exact gate0/first G copies;strict-prefix G replay);4 aggregate F admission
>N/C/M/B/B0,whole/local>N/C/M/B/.5,Brier<N/C/M/B/B0/prior;1 majority-date F−B
admission;8 both-scheme F−B admission/whole/local positive and Brier upper<0;
12 both-scheme F−N/F−C/F−M whole/local positive;6 both-scheme F−N/F−C/F−M admission
positive. Keep all failures; equal first-eligible bases and immutable control/
preprocessing replay are mandatory integrity requirements beyond these gates.

Useful accumulated-evidence value needs timing gains over current AND anchor
controls under both schemes with admission/error checks also passing. Similarity
or losses leave this memory addition unsupported. No adoption of C/N as a fallback,
gate/τ expansion or stronger memory constraint after results. Passing development
checks still authorizes no model/policy promotion or live trading.

## Execution requirements

Before fitting, pin all source/input/artifact digests, authenticate official328
and reproduce full/fold memory/gate/offset fingerprints. Test episode reset,
first-eligible equality, update-at-current-clock causality, last-eligible elapsed
time, censored updates, gate0 carry, future/prefix/label invariance, shared frozen
numeric transforms, convex finite-difference gradients and original weights.
Full tests and critical Ruff first.

Save original ledgers/pairs, observed evidence states/designs, targets/weights,
corrections/logits and complete full/fold fit audits. One cached official replay,
source/digests authenticated, all numeric fields/coefficients/tables<=1e-9 with
exact identities/missingness/tie-aware ranks. Write large parquet tables through
temporary files, verify readable complete footers, then atomically rename; the
R328 local truncation motivates this operational check. Preserve every failed
attempt and do not relax checks. Retire the one-shot workflow after closeout.
No raw acquisition, sealed-date access, main merge, policy change or promotion.
