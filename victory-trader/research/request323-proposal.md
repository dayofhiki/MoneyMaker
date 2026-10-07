# Proposed R323: jointly learn absolute admission and relative state timing

This is an unexecuted proposal, not a completed preregistration. Freeze exact
inputs/feature union/solver/weights/checks before any new fit or evaluation.
R321 supports nearby timing beyond an elapsed-only control. R322's separately
trained meta-classifier did not transfer that signal into absolute admission.
Keep its15 failed gates and R320's failed incremental lag criteria. Do not select
V or another retrospective winner as fallback.

Next mechanism: test whether the absolute state-classification objective weakens
within-episode timing, and whether one fixed jointly trained score can retain
both. Use only pinned official cached May trajectories, unchanged causal features,
labels/costs/stops and seals. No static interactions, new clocks or delay search.

A candidate shared linear logit has coefficients beta and absolute intercept b.
Absolute logistic loss uses ALL8792 complete training states/336 episodes with
original independent weights. Relative logistic loss uses ALL47825 positive-
negative pairs/50 mixed episodes with inherited independent pair weights, and
the score difference beta*(x_positive-x_negative); b cancels. Normalize each
objective to equal declared effective mass before a fixed1:1 mixture. Keep the
same C=.1 regularization scale and no penalty on b. Specify the exact numerical
optimization/tolerances and verify against ordinary logistic regression for
the state-only boundary before fitting real data. No mixture-weight search.

Use one shared preregistered causal feature allowlist and SAME all-state training
transform for joint J and state-only matched C. Include a matched Q/history-only
joint control to test directional price/activity information. Preserve frozen
R320 W for relative-rank reference. The pair component changes the learning task,
not inference inputs: each decision needs one current causal state only. Output
sigmoid shared logit only if the absolute target/calibration checks support it;
never treat the pair-only W score as an absolute probability.

Report all655 complete evaluation episodes and828 identities, primary between-
episode same-day admission rank, Brier versus matched classifier/prior, conditional
timing rank and inherited30s local-pair rank. Require independently positive
cluster intervals for BOTH incremental admission and timing against matched C,
plus Brier/prior and support criteria; predeclare exact gates, seeds and all
failed checks. Fixed-fit intervals still omit model-fitting uncertainty.

This proposal tests an objective mismatch; it does not establish that mismatch
caused R322's failure. Even a pass on reused May is development evidence only.
Separate executable WAIT/ENTER/HOLD/EXIT profitability and independent validation
remain later contracts. No automatic promotion or sealed-date access.
