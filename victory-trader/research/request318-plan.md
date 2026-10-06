# R318: fixed nonlinear and joint-signal interpretation on R317 support

Registered before new fits. Stacked on independent draft PR90, remote parent
927de5f80291c91bed8b635da49a743293785772. R316/316B are separate event tracks.
May is reused development, including known R317 evaluation. June HOLD and
July-August remain sealed; no market requests, new labels, policy or promotion.

## Frozen inputs

Use official R317 run37472421963/artifact11417253645, source
790b831683ce00dc4d3a4f54dab4db8eead10feb, and original canonical first states
from official R315 run37450538746. request318-inputs.json pins every read input
SHA256 before fitting. Verify hashes, fixed dates, manifest, broad identities,
labels/missingness and complete first-state replay. Do not regenerate or tune
clocks/labels. Only R317's8,792 complete continuous states from336 episodes
enter new fits; all9,325 states/407 ledger episodes stay visible. Evaluation is
the SAME828 identities/679 observable scores/650 complete first labels.

## Preregistered model matrix

| Arm | Inputs | Fixed learner | Role |
|---|---|---|---|
| R |saved original frozen score|unchanged R310 M|original benchmark|
| C |M28|unchanged linear pipeline, seed20265105|R317 replay|
| Q |Q9|same linear pipeline/rows/seed|R317 replay|
| X |M28|100 depth2 boosting trees, max4 leaves|primary joint-capable head|
| K |Q9|identical depth2 learner/rows|capacity-matched control|
| A |M28|100 depth1 trees, max2 leaves|additive nonlinear logit comparator|
| D |Q9|identical depth1 learner/rows|additive nonlinear control|

All new heads: sklearn GradientBoostingClassifier(loss='log_loss',
learning_rate=.04,n_estimators=100,subsample=1,criterion='friedman_mse',
min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=.025,
max_features=None,random_state=20265205,n_iter_no_change=None).
Depth and leaf counts are as declared; other defaults remain untouched.
No model, seed, feature, threshold, leaf-size, learning-rate or iteration search.
Report complete parameters for provenance. No selecting A/D/K as fallback.

Each arm uses identical independent_weights and training-only fit_transform
from R317: weighted1/50/99% clipping/imputation, centering/scaling and explicit
missing flags. C/Q preserve fit_linear exactly. Effective weight sums to336,
equal day mass and equal episode mass within day. The leaf minimum is2.5% of
TOTAL fitting weight, not75 correlated seconds; report actual leaf weight and
distinct episode support by assigned training leaves. It is not a guarantee of
out-of-date generalization. No resampling seconds as independent examples.

X/K allow pairwise paths; A/D trees split only one feature, yielding additive
univariate nonlinear logit scores. X-A changes tree complexity, not a perfectly
isolated interaction treatment. Capacity-matched K and difference-in-increments
help distinguish a generic learner effect from directional increment. Joint
paths can be control-control or signal-signal, not necessarily context-signal;
record path-type counts including missing flags, never causally interpret impurity
importance or select features from them.

Score all available cases independent of labels. Require C/Q probability replay
within1e-9 versus official R317. R is copied unchanged, not refit. Chronological
diagnostic: May6/7/8 fits only preceding May5-8 continuous training states and
scores original held-day broad canonical first observations; no current/later-day
transform or label use, no May5 score without earlier history. Replay original
R317 chronological C/Q probabilities within1e-9. No evaluation-label fitting.

## Metrics and fixed decision

Report same-day AUROC(primary), pooled AUROC/AP/Brier, own training-prior Brier,
all-date scores, support/coverage and chronological metrics. Fixed-score paired
day and ticker-day bootstrap1,000 draws, seeds20265250/20265251. Contrasts:
X-C, X-K, X-R, X-A, A-D, K-Q. Also report difference-in-directional-increment:
(X-K)-(C-Q) and (X-K)-(A-D), for all four metrics. Count invalid rank draws;
intervals condition on fitted models and exclude fitting uncertainty.

Fixed checks: coverage>=90%; >=20 positives on>=4 dates; X same-day AUROC/AP
beats C/K/R; X Brier beats C/K/R and its own prior; X-C AUC improves a majority
of evaluable dates; ticker-day same-day lower CI>0 separately for X-C,X-K and
(X-K)-(C-Q). Secondary X-A/A-D and joint-path counts remain descriptive; never
select the best learner or relax a failed criterion. Even all-pass is development
only and cannot establish executable returns or open the sealed data.

Persist hashes, score/chronological checkpoints, parameters, fit transforms,
weight/leaf support and path audits, all comparisons/gates. Synthetic tests cover
scope/membership, matched inputs/weights, unseen-data transform invariance,
single-class fallback, stumps/no joint paths, weighted leaves, chronological
scope and saved-score integrity. One branch-only official cached run; compare
all JSON numbers within1e-9, identities/labels/missingness, pair ranks and flags
against local output. Retire the one-shot workflow after success; main and
research-request.json remain unchanged.
