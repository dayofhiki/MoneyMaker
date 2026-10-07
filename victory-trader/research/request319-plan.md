# R319: causal state evolution versus snapshot interpretation

Registered before constructing evaluation clocks/labels, trajectory features or
new fits. Parent draft PR91, SHA92e7d10e7c63d8e2da9fc44e4a3e457703fb64c0.
R316/316B remain separate. All May evaluation is reused development; June HOLD
and July-August stay sealed. No acquisition, policy, tuning or promotion.

## Fixed support and causal construction

request319-inputs.json pins official R315 canonical first training, official
R317 continuous states/manifest/ledger, official R318 first scores/summary,
and the original R311 cohort/raw history. Verify all hashes before any work.
Retain the same828 evaluation identities. Enumerate EVERY completed regular
aggregate whose start>=HOT, end<=HOT+300s and end<session/HOT+1h cap.
Persist the outcome-independent manifest and full episode ledger BEFORE labels.
No forward filling, winner-selected observation, best delay, or max-score entry.
Use original M28/Q9 features with background frozen at HOT. Unchanged next-open,
BASE cost, absorbing -2.5% stop, HOT+1h/session label. Censor unresolved states;
replay all available canonical first features and label fields versus R318.
First-label censored episodes are not dropped from later observations.

## Fixed learner matrix

Reuse8,792 complete states/336 May5-8 episodes. Construct history features on
ALL9,325 observations before complete-label filtering. Same rows/independent
weights and original training-only transforms for every head. No state labels
or scores enter history features. Total fitting weight remains336 episodes.

| Arm | Inputs/learner | Purpose |
|---|---|---|
| C/Q | original linear M28/Q9, seed20265105 | snapshot replay |
| X/K | original R318 depth2 M28/Q9 | fixed passive nonlinear reference |
| S | M28 + history support, original linear seed20265305 | elapsed/support control |
| G | Q9 + same history support, original linear seed20265305 | direction-free control |
| T | S + seven fixed causal lag differences, same linear seed | primary trajectory head |

History support: seconds since first post-HOT observation, log1p(number of
completed post-HOT observations so far), elapsed age of the latest observation
at or before current clock-30s. Missing lag stays missing; never use nearest
future observation. Seven differences=current minus that lagged state for
30s return, efficiency, volume/transaction rate ratios, signed-volume proxy,
prior-high reclaim, and 10/30 acceleration. Differences have an elapsed-age
support feature but are not divided by elapsed time. 30s is inherited from the
feature package, not a searched lag. C=.1, lbfgs/tol1e-8/max_iter2000 and missing
flags remain fixed. T-S tests this one trajectory package conditional on current
M and observation support; adding dimensions changes capacity, so it is not a
perfect isolated causal order intervention. T-G is the full direction increment.
S-C separates timing/support from explicit lag differences. No fallback choice.

Score ALL observable evaluation states independently of label availability.
Replay C/Q/X/K at first clocks within1e-9 against R318. Broadcast each arm's
own canonical first probability as arm0 to all states of that episode. Compare
dynamic and constant-first scores on IDENTICAL current-state labels and weights.
Preserve original650 complete first-label metrics separately, including R anchor.
Do not compare all-state AUROC to first-state AUROC as an identified improvement.

Chronological diagnostic: May6/7/8 use ONLY preceding complete training states
and transforms, scoring every held-day clock. No current/later-day fit. It has
different support from May11-20 evaluation and is not independent confirmation.

## Metrics and fixed decision

Equal day mass, equal episode mass within day, equal complete-state mass within
episode. Report pooled AUROC/AP/Brier/prior error; primary SAME-day BETWEEN-
episode AUROC excludes same-episode pairs, and within-episode AUROC is an
equal-day/episode conditional diagnostic among episodes with BOTH labels.
Also report within-episode paired Brier errors on all complete episodes.
No score maxima or profit claims. First/censored/later-complete and ever-positive
episode counts remain visible. HOT-elapsed bins (0,30],(30,120],(120,300] and
lag-available/unavailable are descriptive fixed partitions, never entry delays.

Paired whole-day and ticker-day multiplicity bootstrap,1,000 draws,
seeds20265350/20265351. Apply sampled multiplicities to ORIGINAL evaluation
weights, retaining each episode's identity; exclude same-original-episode pairs.
Report valid/invalid rank draws and95% intervals for T-S,T-C,T-G,S-C,
T-T0,C-C0,X-X0. No resampling seconds. Intervals omit fitting uncertainty.

Predeclared hypothesis checks: >=90% episode coverage with any complete state;
>=20 ever-positive episodes across>=4 dates; T beats S,C,G in primary rank/AP
and Brier, and Brier beats own prior; T-S rank improves a majority of dates;
positive ticker-day AND day lower rank CI for T-S and T-G; T within-episode
AUROC>.5; ticker-day AND day lower rank CI>0 for T-T0. Keep every failed check.
All-pass only supports this development representation, never policy adoption.

Synthetic tests cover future/label invariance, exact sparse lag selection, resets
between episodes, history before censor filtering, no pseudo-clocks, scope,
first replay, constant-first broadcasts, same-episode-pair exclusion, weighted
rank equivalence, cluster multiplicities and chronological exclusion. Persist
all states, manifest, ledger, first anchors, training trajectory and fit audits.
One branch-only cached official execution; reproduce all JSON values, clocks,
labels/missingness and within-episode probability ordering within1e-9. Retire its
workflow after success. Main/research-request.json remains unchanged.
