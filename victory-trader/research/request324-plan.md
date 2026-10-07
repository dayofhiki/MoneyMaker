# R324: recursive latent opportunity from causal observations

Register remotely BEFORE real-data transition fits/evaluation. Parent official
R3236537b156400a81fd521a5351bdc83866bb247e83/Draft PR96. R323's shared logit
traded admission for timing; test a distinct state-evolution mechanism. Routine
cached research/publication has standing user authorization. This follows its
recorded R324 proposal and TRAINING-ONLY support audit, not an evaluation search.

## Frozen scope and transition targets

Eight exact official R319/R321 files are SHA256 pinned in request324-inputs.json.
Same9325 original training observations/8792 complete states/336 episodes,
828 evaluation identities/19530 observations/18793 complete states/655 episodes,
78 mixed timing and70 local-pair episodes/all8623 inherited30s pairs. May remains
reused development, not fresh validation. No acquisition, clocks/labels/costs/stops,
sealed June HOLD/July-August, promotion, live trades or main configuration change.

Sort ALL original observed states by clock within original episode; compute new
causal state_previous_t and state_gap_log_s=log1p((t-previous_t)/1000), missing at
first observation. Actual gaps include up to240s, not assumed regular1s. No forward
fill or bridging past missing observations. Censored observations stay in history.

Enumerate ONLY adjacent ORIGINAL observed endpoints when BOTH labels are complete;
never skip a censored observation to create a training transition. Hindsight latent
y_t is the unchanged reachable BASE net>=5 target, not an online observable or
realized profit. Persist all8445 complete adjacent transitions/286 episodes before
fits:0→0=7553/273 episodes,0→1=88/38,1→0=99/48,1→1=705/45. Conditional previous0
support7641 rows/279 episodes; previous1 support804/54. Independence is by episode,
not transition count. Prior labels condition supervised TRAINING heads only.

## Four fixed recursive arms and matched initial state

For each representation fit q01(x_current)=P(y_current=1|y_previous=0,x_current)
and q11(x_current)=P(y_current=1|y_previous=1,x_current), using CURRENT causal
features. Previous label is used only to select training rows, never as a current
feature or evaluation input. No teacher forcing/reset using true held labels.

| Arm | Transition representation | Role |
|---|---|---|
| F | R319 T38 + state_gap_log_s | primary rich state evolution |
| H | R319 G12 (Q9+history) + same gap | matched context/history control |
| A | R319 three HISTORY inputs + same gap | matched elapsed/gap control |
| K | conditional training priors q01/q11 | constant transition control |
| B | frozen saved R319 G current classifier | unchanged absolute baseline |
| B0 | frozen G own first score on same later labels | initial-only baseline |

Shared initial p_first for EVERY recursive arm is frozen R319 G's current first
probability. Reconstruct its saved coefficients/transform and replay G probabilities
within1e-9. Do not choose another initial model after seeing results. Each arm
shares the same initial-score/prior and inference clocks.

Fit each feature transform on ALL8445 transition-current states/286 episodes,
using independent equal day/episode/state weights and existing1/99% clipping,
weighted median/mean/scale/missing flags. Copy that transform to BOTH conditional
heads. Supervised weights are recomputed separately within previous0 or previous1
support, equal day/eligible-episode/transition mass, sums279 and54. No balancing of
rare transition classes, oversampling or pair mass inflation. Conditional learners
LogisticRegression(C=.1,l1_ratio=0,solver=lbfgs,fit_intercept=True,max_iter2000,
tol1e-8,random_state20265805); convergence warning is an error. Single-class
nonempty conditional support returns its own empirical prior. Empty conditional
support leaves the arm unscored after its first initialization, with held rows
retained and explicit ledger; never fabricate an unseen-state probability.

At each subsequent observed completed clock propagate PREDICTIONS ONLY:
p_t=(1-p_previous)*q01(x_t)+p_previous*q11(x_t).
The previous probability already exists, every q input exists now, and no true
past/current/future label, censor flag, future mean or peak enters inference.
Process ALL observations, including censored labels. Do not clamp/threshold/reset,
select future-best time, or learn a regular-time interval assumption. This is an
approximate latent-transition contract, not proof of an identified Markov process
or optimal Bayesian filtering; prediction error can compound.

Chronological diagnostic: May6/7/8 transition heads/transform use ONLY original
preceding training days. Initialize from the matching frozen R319 G preceding-day
fold model; replay saved official R321 chronological G scores within1e-9. Keep all
7806 held observations; diagnose unscored support explicitly. No own-day fit or
held label enters recurrence. Synthetic future/label/censor mutation tests and
causal-only inference without outcome columns must pass before real fits.

## Metrics, intervals and24 gates

Primary SAME-day BETWEEN-episode AUROC on ALL complete evaluation observations,
excluding same-original-episode pairs. Original equal day/episode/state weights.
Report pooled AUROC/AP, Brier vs common frozen initial model's training prior,
conditional whole timing AUROC and inherited R321 local30s pair rank with its
original eligible day/episode/pair weights. All first predictions are identical;
report first baseline once, never claim first-clock improvement. Describe full
per-day metrics and fixed long-gap diagnostic (>30s vs<=30s; no delay selection).
No entry threshold/policy or realized returns. Preserve frozen W whole/local timing.

Bootstrap1000 whole day/ticker-day multiplicities on ORIGINAL weights, seeds
20265850/20265851. Compare F-H,F-A,F-K,F-B,F-B0,H-A across all six metrics. Pairs
are not independent; intervals condition on fits and omit both stages' uncertainty.

Twenty-four checks: >=20 training onset episodes,>=20 decay episodes, both onset
and decay across>=3 training dates,>=200 transition training episodes,>=50 mixed
evaluation episodes,>=30 local evaluation episodes,>=4 positive evaluation dates
(seven support checks). F admission>H/A/K/B/B0; F timing>H/A/K/B/.5; F local>H/A/K/B/.5;
F Brier<H/A/K/B/B0 and common prior (four). F-B admission improves on a majority
of dates (one). Separately positive DAY/TICKER-DAY admission, whole timing and
local CI for F-B (six), positive admission CI for F-H (two), positive admission
CI for F-A (two), negative upper Brier CI for F-B (two). Keep every failure; no
arm/threshold/feature/solver/gap search or fallback selection. Even all-pass is
development evidence, not model/policy promotion or access to sealed data.

Persist training transition manifest/ledger, all enriched training/evaluation/fold
states, q01/q11 and recursive probability checkpoints, fits/weights/transforms,
input hashes and828-identity ledger. Full tests and critical Ruff; one official
cached reproduction after local validation; compare JSON/values/missingness and
all probability-score ordering within1e-9. Verify artifact digest, retire one-shot
workflow, and record honest outcomes including all unsupported/no-fit folds.
