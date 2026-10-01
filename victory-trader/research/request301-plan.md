# R301 preregistration — preserve chart context across entry

Status: implemented for one consolidated research dispatch. 43 tests and research
lint passed; full local execution reproduced R300 C exactly and completed every
nested fold with all 86 positions reconciled. The local economic gate failed.
No parameter, feature-definition or policy changes followed that result. The
design below was fixed before R301 results. May5-8 development only; June sealed.

## Hypothesis and controlled arms

R300 C's extra 60s context was unavailable at 69/72 model exits because its
feature prefix began at entry. Existing raw regular-session seconds contain
pre-entry observations for all 86 entries. Test whether retaining that causal
history helps initial HOLD/EXIT decisions. Do not impose a minimum holding time.

A: exact R300 C replay using saved OOF predictions.
B PRIMARY: same 28 feature names, model/hyperparameters/weights/seeds, pi0 and
two-stage nested policy-improvement algorithm, but the eight R300 clock/context
features use all saved completed regular-session bars before AND after entry.
The original 20 position features remain unchanged and entry-relative.

Same 86 entries / 97 candidates, R296 entry gates, -2.5% BASE-net stop, costs,
caps, absorbing pending orders and next-regular-open reference fills. No entry
filters, new features, extra policy iterations, threshold grids or stop widening.
Dependencies R300 run 36830074540 (states/decisions) and R298 run 36801235949
(raw seconds/old reference decisions). Zero market API requests or new dates.

## Full-prefix feature availability contract

Use only regular-session bars whose end <= decision_t, sorted and unique. No
unfinished/future bars, interpolation, synthetic silent seconds or future fills.
The oldest observed completed bar is a conservative history boundary, not a
claim of dense coverage or guaranteed quote availability. Do not use entry-price
anchor as a replacement for an unavailable pre-entry close.

- Return w=5/20/60s: current completed close / latest completed close with
  bar_end <= t-w, minus one, times100. If no such close exists, NaN.
- Count latest 20/60s: completed bars with t-w < bar_end <=t. Require a known
  completed bar at/before t-w, otherwise NaN. Gaps are counted as absent bars.
- Volume ratio compares (t-20,t] with (t-40,t-20]. Require a known completed
  bar at/before t-40 and positive prior-window volume; otherwise NaN.
- 60s close location/drawdown use completed closes in (t-60,t]. Require history
  at/before t-60 and at least one window close. Flat range location=0.5.
  Drawdown 100*(current/max_recent_close-1); location (current-min)/(max-min).
- Use the model's existing NaN handling; retain all positions and forced stops.
- Pre-entry history may precede HOT if present in the saved artifact, but cannot
  cross regular-session/day/ticker boundaries. No additional acquisition.
- Preserve full position prefix after hypothetical voluntary exits. Risk
  reachability still ends at first mandatory stop, independent of these features.

## Training and evaluation

Reuse R299/R300 nested day exclusion and fixed seeds, with B's own pi1 downstream
labels. Report target differences; they are not identical labels across policies.
Fit/test days remain May5-8 only. 10,619 reachable rows represent 75 trajectories
from 86 total positions, not independent training examples at that row count.

Report mean/median/daily BASE, LIGHT/STRESS, gross/cost drag, CVaR5, positive
positions, concentration, hold durations, first-decision exits, delay counts,
model-exit feature availability and first-state feature availability. All 86
reconcile, including eleven without first reachable HOLD and nine cost-only stops.
Use both day and ticker-day cluster uncertainty for B-A and B-R299 D differences.

Separate own-target rank from common frozen-pi0 and saved A-pi1 target ranks.
For common labels show all-reachable, first-reachable and actually visited states,
prediction/target signs, confusion counts and conditional mean advantage. These
are exploratory policy-dependent counterfactual diagnostics, not optimal labels.
Do not infer a better threshold from their results in this run.

Primary B must meet all gates: exact A replay within1e-9; all86 reconciled;
>=95% resolution; positive BASE mean; positive on>=3/4 days; positive paired
gain vs A, R299 D and old trailing; positive mean excluding largest winner;
positive day/episode-weighted ranked OWN continuation advantage.
Common-label diagnostics qualify interpretation but cannot override failed
economic gates. Reused four days and upstream entry OOF prevent promotion.

## Tests and boundary

Test pre-entry history available at first second, exact window ties, sparse
coverage/zero volumes, missing history without entry-price fabrication, unfinished
bars, future mutation invariance, group/session boundaries and convergence to
the R300 post-entry definitions once each complete window lies after entry.
Retain inherited risk/cap/nested provenance and exact replay checks.

If B fails, do not keep adding chart features or open June. Next distinguish
unstable continuation-policy targets/action calibration from entry/cost economics
in a separately preregistered diagnostic. If B passes, freeze before additional
later-May development; those dates have previously been explored and are not a
pristine holdout. No selection or retuning after this run's outcomes.
