# R307 official findings — calibration cannot rescue unstable early signal

Official run37212562599 completed successfully; source7f6e9d6bfb10a41bc27404e9c0e299824fc615ad, artifact11307291571. Primary metrics and all checks reproduce the preregistered local result within1e-9. Exact R306
C scores reproduce with zero error on all four folds.48 selected tests and
critical lint pass. No feature/model/seed/threshold changes followed results.
Zero market requests, same97 episodes, same4526 states, June sealed.

First-snapshot raw C AUROC0.597127, AP0.127366, Brier0.103026. Calibrated E
AUROC0.462485, AP0.091669, Brier0.088424. Cost-only D Brier0.100021;
FIRST-snapshot training-prior Brier0.084378 still beats E. Original state-prior
benchmark0.081417 is separately reported, not mixed with first-prior calibration.
E-C Brier difference95% ticker-day CI[-0.029192,-0.002486] shows a real reduction
in this loss on these reused days, but does NOT show improved discrimination.

Held May5/6/7/8 calibration slopes0.333690/0/0/0.347991. All optimizer runs
succeeded; zero slopes are optimum constant maps, not a code/optimization error.
They cannot preserve within-day rank. Primary gate FAILED: E loses raw C rank,
fails to beat first-prior Brier, and two folds become constant. Do not deploy E,
force positive slopes, invert scores, adjust thresholds or retune regularization.

Raw C weighted first-snapshot mean probability0.047866 versus observed0.089182;
E mean0.095493. The earlier colloquial claim of probabilities being generally
"too high" was incorrect. Bad probability QUALITY involves wrong high/low
confidence for individual candidates, not necessarily an inflated mean.

## Refined bottleneck

R306 showed a modest candidate-ranking gain from risk/cost-aligned targets and
clock momentum. Nested R307 prediction/calibration does not establish stable
cross-day early probability signal. There are only9 independent first-snapshot
net+5 winners in97 selected candidates (May5 none), although4526 correlated
state rows can make the nominal training size look large. This is insufficient
support to validate a99-feature rare-tail learner across regimes. That is a
support/generalization diagnosis, not a proof momentum is inherently unknowable.

The next research priority is additional INDEPENDENT early momentum examples,
with the current99-feature representation and stop/cost contract frozen for
reference. Expand a preregistered causal May development candidate population
BEFORE the strict learned shortlist, preserve all original97 as an evaluation
slice, and split by day/ticker trajectory. Determine whether actual sustained
flow/price-pressure patterns generalize, using low-dimensional clock feature
families as controls. Never duplicate seconds as extra independent winners,
select examples by future spikes, relax stops/costs or open June for tuning.
No R308 has been dispatched here; its population/data-acquisition scope must
be specified before collection/fitting. Preserve original entry/exit references.


First-snapshot raw C probability>=0.5 group contains4 episodes and zero net+5
winners; all9 winners lie in lower-scored groups. These sparse descriptive bins
explain why a modest overall rank gain does not justify using probabilities
as betting conviction. No inverse or mid-range threshold is selected from them.

A further hypothesis to isolate before attributing the failure solely to data
quantity is early-versus-later training distribution: current episode weighting
balances trajectories but lets numerous later cash states dominate each one.
Future research should compare preregistered time-balanced/early-snapshot
supervision and constrained low-dimensional pressure features, alongside added
independent causal candidate coverage. Reweighting is not extra independent data.
