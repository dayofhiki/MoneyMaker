# R318: shallow joint interpretation does not repair first-observation momentum ranking

R318 is stacked on draft PR90/R317. Preregistration remote386b8c71b9530f103d014fd435df2b9d57edd11d
and local888530c preceded new fits. The declared chronological checkpoint hash
was added before fitting, without changing any model or comparison. R316/316B
remain separate event studies. Main and research-request.json remain unchanged.
All May results are reused development, including previously known evaluation;
June HOLD and July-August remain sealed. No new acquisition, labels or policy.

## What was tested

Use exactly R317's8,792 complete continuous states from336 broad May5-8
episodes, preserving all9,325 original clock states and407 ledger episodes.
The same828 evaluation identities,679 observations and650 complete first-clock
labels are retained. Pinned hashes, manifest/membership, first-state features,
labels and missingness verify. Unresolved labels never become negatives.

C/Q replay the original linear M28/Q9. X/K use100 fixed depth2 boosting trees
with maximum4 leaves; A/D use100 depth1 trees with maximum2 leaves. X/A include
the same directional package, K/D remove it. New learners use identical
training-only weighted transforms, independent weights, fixed seed/parameters,
and a2.5% fitting-weight leaf minimum. R copies the original frozen M score.
No hyperparameter, learner, feature, threshold or observation-window search.

The effective training weight remains336, not8,792 independent examples.
In the full fit, X's smallest leaf has8.4076 weight and10 distinct episodes;
K/A also have at least10, D at least24. X actually uses joint paths:
158/382 leaf paths combine a control and a signal-package variable,184 are
control-only and40 signal-only. A/D have no mixed paths. Signal-package paths
include missing indicators. Counts show the declared model has used joint
inputs; they are not causal feature importance or proof of useful interactions.

## Same650 final-label development comparison

| Fixed arm | Same-day AUROC | Pooled AUROC | AP | Brier |
|---|---:|---:|---:|---:|
| R original frozen M |.640588|.642512|.099204|.056287|
| C continuous linear M |.609553|.603009|.085894|.058848|
| Q same-row linear controls |.639917|.630069|.142359|.056895|
| X joint-capable M |.603439|.597089|.090816|.060080|
| K same-depth controls |.594763|.586288|.089518|.060459|
| A additive nonlinear M |.624993|.628102|.094201|.057475|
| D additive nonlinear controls |.612806|.611293|.109969|.057117|

All broad arms have own-prior Brier .056820; none beats it. A is a fixed
secondary comparator, not a chosen fallback or promoted learner.

X-C same-day difference-.006114 has ticker-day95% CI[-.059784,+.052223]
and day CI[-.036574,+.019939]. X beats C on only four/eight dates. Its AP
point difference+.004922 is also inconclusive. X Brier is worse by+.001232:
day CI[+.000193,+.002401], ticker-day CI[-.001255,+.003508]. There is no
robust primary improvement.

X-K directional increment is+.008676 same-day AUROC, with ticker-day CI
[-.031425,+.053450] and day CI[-.039400,+.047304]. AP increment is only
.001298 with intervals also crossing zero; Brier difference-.000379 is
inconclusive. More flexible same-row interpretation does not establish a
directional edge. It does not prove such an edge is absent in other contracts.

X-A same-day difference-.021554 has rank intervals crossing zero. Probability
error is clearly worse on these fixed scores: Brier difference+.002605,
ticker-day CI[+.000510,+.004624] and day CI[+.000956,+.004156]. Increasing
tree complexity is not sufficient; X-A also changes capacity and is not an
isolated market causal interaction experiment.

The difference-in-directional-increment versus linear is+.039039 same-day
AUROC, ticker-day CI[-.010488,+.096415] and day CI[-.018290,+.101980].
Its AP difference+.057763 has positive intervals (ticker-day[+.009511,+.136544],
day[+.012967,+.131024]), but this is primarily the nonlinear CONTROL head
losing AP: K-Q is-.052841, with negative AP intervals under both schemes.
X-C AP improves only+.004922 with intervals crossing zero. Thus a relative
AP increment is not an absolute successful joint head. Difference versus
additive directional increment is inconclusive across all reported metrics.

Only2/10 fixed checks pass: positive episode count and positive-date support.
Coverage remains650/828=78.50%, and every rank-improvement/material comparison
fails. Both bootstrap schemes have1,000 valid rank draws; fixed-fit intervals
omit fitting uncertainty. No primary/secondary model or trading policy is adopted.

## Chronological training diagnostic

May6/7/8 use preceding continuous states only; no current/later-day transform,
label or fit. On the same255 complete held first states, same-day AUROC is
C .789351, Q .800464, X .736447, K .725391, A .717089, D .749057.
C/Q replay original R317 chronological scores within1e-9; maximum C difference
9.33e-11, Q8.33e-17. Evaluation baseline replay errors are at most1.70e-15.
These chronological cases/fits differ from the650 evaluation and are not an
identified temporal deterioration estimate or independent confirmation.

## Research decision

R317 restored continuous training phases; R318 verifies that adding fixed shallow
joint interpretation is not a sufficient repair at the original first evaluation
clock. Both direction and observation activity may require information not
present at that clock, better input representation, different support, or other
model/risk contracts. R318 does not identify which explanation is correct.
Do not respond by searching deeper trees/thresholds on these known650 outcomes.

The next separately registered diagnostic should retain the same full evaluation
cohort/raw history and examine ALL completed causal observation states in the
fixed inherited early window, using frozen training-only models. Preserve the
original first-clock comparison as an anchor; preregister any phase descriptions,
day/episode weights and cluster resampling. Never substitute each episode's best
future state, report max-score profit, choose a winning waiting delay, or treat
correlated seconds as independent successes. No evaluation refit, stop/cost
change, sealed date or model promotion follows this proposal.

That study addresses a specific unresolved distinction: first-clock failure
versus failure to interpret information during continued monitoring. It would
still diagnose alternative cash decisions, not demonstrate sequential realized
trading returns. Any later operational policy needs its own evaluation.

## Official closeout and reproduction

[Official run37476099213](https://github.com/dayofhiki/MoneyMaker/actions/runs/37476099213)
**succeeded**, source431b860a148bdf40b34cfe43481a9f70bf91823c,
artifact11419038296. ZIP SHA256 verifies:
`3afbc7eb93681b2be6ee821bcd5f8f0d5621690e8f2eea29193af9c03b83f376`.
All3,982 JSON numeric fields and all nonnumeric values agree exactly; official
and local JSON bytes are identical. Both scored output tables (828 evaluation,
305 chronological rows) agree exactly on values, identities, clocks, labels,
missingness and complete probability pair ranks. Offline replay has zero requests.
Exact official/local JSON and request318-reproduction.json record the evidence.

Corrected source full GitHub CI: **1,051 tests pass**, with both test and
flatfiles-access jobs successful. Official and local targeted validation:
**64 tests pass**, critical Ruff passes. The first run37475904829 failed in
reporting because legacy metrics reserved arm A for a nonprobability score;
the newly registered A is a probability head. Generic probability metrics plus
a regression fix this name collision without changing fits, labels, weights,
parameters, gates or protocol. The failed run is not a different hypothesis.

The one-shot branch workflow is retired after closeout. R318 remains draft and
unmerged; no source model is promoted, and merging cleaned documentation later
will not dispatch a duplicate experiment or alter research-request.json.
