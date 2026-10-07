# Proposed R320: within-episode learning objective for opportunity timing

This is a **next-study proposal**, not an executed or preregistered experiment.
Freeze exact input hashes, parameters, tests and complete criteria in a separate
registration before any fits. R319's failure does not establish objective mismatch
as a cause. The question is whether explicitly learning within-episode ordering
adds timing information that pooled state classification missed.

Reuse R319's fixed training trajectories, all828 evaluation identities, all
observed completed clocks, labels/censoring, costs/stops/caps and sealed periods.
No new raw data, lag/window/threshold search, architecture sweep or policy.

Train a linear pairwise learner on a positive and a negative COMPLETE state
from the SAME training episode. Each endpoint uses its own completed causal
features. Build history before censor filtering. Apply the training-only R319
state transform first, then subtract transformed endpoint vectors. Include both
orientations with complementary labels; no intercept. At inference score ONE
current state's transformed vector with the learned coefficients. Never subtract
a future state, use an episode's future mean, or inspect evaluation outcomes to
choose the states scored.

Use all eligible pairs without hindsight selection of the strongest pair or a
winning delay. Equal day mass and equal mixed-episode mass within day; each pair
shares its episode mass, with total effective fitting weight=50 mixed training
episodes on the inherited support.47,825 pairs are correlated. Persist membership,
class support and pair-weight mass before fitting. Outcomes choose training pairs
by definition; all evaluation states remain scored irrespective of their labels.

Fixed input arms should include Q9+history support, M28+history support, and the
same M+support+seven trajectory deltas from R319. Same learner and pair support
for all three. A fixed C=.1/tol1e-8/max_iter2000/no-intercept logistic pair learner
is a reasonable starting contract to freeze; no C or feature search. Preserve
R319 S/T as saved classification-objective references. Add chronological fits
using only preceding mixed episodes; explicitly retain single-class/no-pair
unscored folds instead of inventing evidence.

Primary within-episode AUROC, with rank improvement against both the snapshot
pairwise arm and saved R319 classification T. Report whole-window between-episode
same-day ranking and fixed phases, but do not promote a cross-sectional ranker
from a conditional timing task. Raw pairwise scores are RELATIVE scores, not
calibrated success probabilities; do not compute a probability Brier score by
applying an unjustified sigmoid. Broadcast each episode's first raw score as a
timing control. Keep all identities/censored states and disclose the conditional
mixed-episode evaluation denominator.

Predeclare whole-day/ticker-day bootstrap and original weights, valid draws,
required support, both-scheme lower intervals>0 for timing increments, and
majority-date consistency. Fit parameters and transforms only on training.
Absolute success probability and a sequential entry/hold/exit policy remain
separate contracts even if the conditional rank test succeeds. A failed result
must not trigger an unregistered recurrent-model or lag/window search.
