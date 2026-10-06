# R314: gross opportunity and net conversion

Preregistered before new stage labels, fits or evaluation. Reused May is
development; June HOLD and July–August remain sealed. No policy promotion.

## Fixed experiment

Use the unchanged R310 May5–8 97-episode, 4,526-state training checkpoint and
R311 outcome-independent 828 May11–20 identities at their original first
observation. Preserve all missing identities. Primary evaluation remains the
650 complete stopped BASE-net labels, not the smaller whole-horizon subset.
Replay original clock features, fills, cost, stop and frozen M probabilities
(absolute tolerance 1e-9). No market acquisition, feature/window/threshold/model
search or earlier May evaluation labels in fitting.

For each observed state use the actual next entry open and the original common
HOT+1-hour/session-close cap, completed-bar submission and pending next-open
fill rules. Derive whole-horizon maxima relative to (1) last completed observed
close and (2) actual entry open, then (3) actual-entry BASE net without absorbing
stop and (4) original stopped BASE net. Stages 1–3 require complete whole-cap
fills; preserve censored rows. The observed-close reference is a counterfactual,
not an executable entry. Repricing can gain or lose opportunities; stages
2→3→4 are nested. Diagnostics use common complete support and disclose its
label-availability selection. No future stage label is an inference filter.

## Models and estimand

All models use unchanged M28 and original fixed linear pipeline, C=0.1,
lbfgs, tolerance 1e-8, maximum 2,000 iterations; transforms fit on each model's
training rows only. R replays original M on all 4,526 states (seed 20264605).
Common training support consists of complete whole-horizon and final labels.
D predicts final stopped net>=5 directly on common support. G predicts
actual-entry whole gross>=5 on the same support. H predicts final net>=5
conditional on gross>=5, using only qualifying **training** rows and their
original common-population weights. Do not rebalance/renormalize conditional
weights. D/G/H seed 20264905. P=G×H for every observed evaluation row, without
evaluation-outcome conditioning. Single-class models use weighted priors;
empty conditional support uses H=0. Report the weight masses and verify
G-prior×H-prior equals the direct net prior. No selected fallback head.

## Fixed comparisons

Primary P versus D on unchanged 650 final labels; R is the original-support
reference. Report day/episode-weighted same-day AUROC, pooled AUROC/AP/Brier,
constant training-prior Brier, per-day results and paired 1,000-draw day and
ticker-day intervals, seeds 20264950/20264951. Resample fixed scores, so these
intervals exclude fitting uncertainty. Component gross/conditional metrics are
diagnostics for different targets/populations, not net-policy improvements.

Gate checks: coverage>=90%; >=20 positives; >=4 positive days; P same-day AUROC
beats D and R; P AP beats D; P Brier beats D and its training prior; P versus D
AUROC improves a majority of evaluable days; ticker-day P−D same-day AUROC
interval lower bound >0. No threshold relaxation; all results remain development
and promotion-ineligible even if these checks pass.

Save all 828 scored identities, training stages, missing/censored counts,
stage/repricing transitions, model fit audits, provenance hashes and gates.
Tests cover pending fills, censoring, nested cost/stop events, repricing gains
and losses, preserved conditional weights/prior identity and scoring independent
of evaluation outcomes. Official CI run must reproduce the local result before
any subsequent experiment is designed.
