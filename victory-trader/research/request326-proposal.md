# R326 proposal: train the deployed recursive trajectory, with matched controls

R325 passes9/24 gates, with whole timing .368617 versus current G .532915 and
worse Brier. Its fixed continuous-time kernel and numerical optimizer are
verified, so another rate/gap/feature search is not justified by these results.
Keep the architecture and test a different learning objective: complete-state
sequence likelihood under prediction-only unfolding. This is a proposal; remotely
register exact numerical/optimization/input/gate contracts before real R326 fits.

## Hypothesis, leakage control and feasible training population

Current adjacent likelihood conditions on true previous training labels; deployed
recursion uses uncertain predicted beliefs. Conditional filtering can be correct
under the right specification. The hypothesis is that approximate dynamics,
finite conditional support and initial-score uncertainty make directly optimizing
the used trajectory preferable. Do not present this as a diagnosed teacher-forcing
bug, assume a correct Markov process or promise improved realized returns.

Use ONLY official R321 chronological May6/7/8 states for NEW rate training. Their
G initial predictions were fitted strictly on preceding days: May5, May5–6,
May5–7. Reconstruct/replay them within1e-9. Do not train recursive loss on G's
in-sample full-training scores. Evaluation still initializes from unchanged full
May5–8 G, exactly as R324/R325. The different training-fold initial fits remain
a limitation, not fresh validation.

Training-only audit request326-training-support.json:7806 observed states in266
observed episodes,7327 complete states in256 episodes,51 positive episodes.
Adjacent complete support7066 transitions/218 episodes,0→1=77 rows/33 episodes,
1→0=85/41, each across3 dates. Previous-zero conditional support214 episodes,
previous-one45. No new model or held-date R326 evaluation has run.

## Fixed learner families to preregister

All recursive models use R325's current-feature continuous-time rate law and
30s numerical rate unit, same shared initial G and actual adjacent observed gaps.
Primary F uses T38, matched H uses G12, age A uses three HISTORY inputs and
constant K two intercepts. No new feature/interaction/gap or architecture search.

For each episode start from its frozen past-fit G first score. Propagate its OWN
predictions through ALL original observations, including censored labels. At each
complete observation accumulate weighted Bernoulli NLL of the current prediction
against the unchanged reachable BASE net>=5 label. Outcomes are loss targets only;
true prior/current labels and censor flags never become recurrent inputs.
Backpropagate through the recurrence using a stable analytic sequence gradient.
First prediction/loss is constant; keep its original weight and loss when defining
total objective mass rather than rescaling after removing its zero gradient.

Complete-state supervision retains equal day/episode/state weights of total256;
L2 slopes at C=.1, intercepts unpenalized. For fair feature preprocessing, fit
transforms on the SAME7066 transition-current states/218 episodes for BOTH new
sequence learners and adjacent-likelihood control, independent transform mass218.
No censored observations are bridged for adjacent supervision; no observations
are skipped in a sequence's prediction history. Do not balance rare outcomes.

M: matched T38 rate model trained by R325's adjacent likelihood on these SAME
training days, same transform/initialization/evaluation and conditional weights
214/45. Include frozen full-data R325 F as descriptive predecessor, but do not
use it as the sole objective control because it had286 transition episodes.

S: matched T38 current-only stationary logistic classifier, initialized to common
G on each episode's first observation, otherwise memory-free. Use same transform
and original complete-state weights; first loss is fixed and has no fitted gradient.
Choose C=.2 BEFORE fitting: the instantaneous two-rate limit has logit beta_a-beta_b
and minimum slope penalty ||beta_a-beta_b||²/(4*.1), matching logistic C=.2.
This static control is needed to distinguish useful recurrence from a rate model
that merely becomes an instantaneous classifier. It is not a new interaction search.

Keep frozen current G/B and own-first G/B0 controls. Require a development gain
over B and both M and S. Record rate/memory-retention distributions; fast-rate
collapse is a possible finding, not grounds to tune bounds or choose a timescale.
No alternative initial model, teacher-forced held-label oracle, new horizon,
entry threshold, class rebalance or secondary-arm selection.

Fix one deterministic training-only switch/exposure initialization, analytic
gradient, stable log-domain probability/loss and optimizer limits/tolerances in
the preregistration; no multistart/retry/search. A failed/nonfinite/boundary fit
is recorded. Preserve every held row and explicit unsupported score scope.

## Chronology, comparisons and decision

For May7 diagnostics train rates only on out-of-time May6 training records;
for May8 only May6–7. May6 has NO preceding out-of-time meta-training day: retain
its rows/initial predictions and report unsupported later rate scores. Do not
fabricate a fit, use own/future-day meta labels, or remove the date from ledgers.
Report common finite diagnostic support separately; it is not the full8-day result.

Main evaluation retains all828 identities/19530 observations/18793 complete states,
78 mixed episodes and70 local episodes/all8623 inherited30s pairs. Require finite
core main predictions everywhere. Same-day between-episode AUROC is primary;
whole timing, local rank, pooled AUROC/AP, Brier/common prior, all dates, first
equality and fixed >30s diagnostics remain. No profit/policy claim.

Preregister1000 whole day/ticker-day draws on original weights/new fixed seeds,
all primary/control contrasts including F-M and F-S. Proposed32 gates: R325's24
roles, with M/S included in aggregate admission/timing/local/Brier comparators,
plus BOTH-scheme positive whole and local F-M intervals (four) and F-S intervals
(four). Merely beating failed R325 is insufficient; preserve every failure.
W .671630 whole/.599223 local remains a frozen descriptive rank reference.

Before real fits: finite-difference sequence gradients across multi-step/censored
histories, future/label/censor mutations, no teacher forcing, frozen initial scope,
same-time composition, matched static-penalty limit, conditional/state weight
independence, unsupported May6, then full suite/critical Ruff. Persist all inputs,
weights/transforms/fits/rates/q heads/trajectories/ledgers; run one authenticated
cached official replay and compare values/missingness/ranks within1e-9, then retire
workflow. Reused May development cannot justify promotion or sealed-date access.
No raw acquisition, main config/cost/stop/label change, June HOLD/July-August,
promotion or live trading. No R326 model has yet been executed.
