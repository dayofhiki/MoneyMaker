# R330: current-state ENTER-now versus causal WAIT policy value

Registered remotely before new fixed-exit labels, fits or policy evaluations.
R329 is closed at281d1745f1ee76bed4f1a25a195208e5a3fcb030:27/40 pass, important
admission/local/error gates fail. Explicit-memory variants are deprioritized.
No memory head is promoted; this is a separate small causal action experiment.

## Question and eliminated shortcuts

Does comparing a cost-aware immediate ENTER payoff with the value of WAIT under
a fixed causal future policy improve recurrent decisions over the same ENTER
payoff alone? Absolute opportunity G, instantaneous evidence and execution cost
are inputs. Within-ticker improvement can affect these current observations;
there is no new stored model memory. Every observed clock can choose WAIT again.

R245 separated tradability from profit but economics failed. R256's average future
ENTER payoff is not a WAIT policy value. R274 prohibits settlement at an earlier
price when an exact deadline is missing. Respect those lessons: never train on a
future maximum, never force unknown execution to cash, never choose an exit
because it makes a label positive. Neither G probability nor AUROC is account EV.

## Fixed support and authenticated caches

Use the exact R3217,806 observed training states on May6–8 and19,530 observations
on May11–20; preserve the original828 evaluation identities. Training G is strict
past canonical G from R328; evaluation G is frozen May5–8 G. R328 current numeric
coordinates, gate and G paths are pinned controls/representation, not selected
fallback models. Exact input digests are request330-inputs.json.

Raw seconds are existing R315 training cache580,944 rows on May5–8 and R311
evaluation cache859,466 rows on May11–20. Authenticate official sources
c271fbab9c3567c470fc852ef80c39fe36116713/run37450538746/artifact11405473272
(ZIP db0efe230c98196ca938eb6b80d41576b5eda017a2f9ef2f62d4dd74fb09f95e)
and a6034adba43fbd02334834f68341a64951957ba6/run37223064559/artifact11311286614
(ZIP dba36fcbace9f0d2d61dc4301b266acca2c2aad1651967adf1e8f77fc72647e4).
No raw acquisition or new dates. Do not use their old policy outcomes as features.
All May results are reused development; June HOLD and July-August stay sealed.

## Fixed downstream ENTER contract

Submit at the original observed clock. Entry is the first regular unadjusted
second open at/after submission, requiring lag<=3s (existing R297 bound). Otherwise
execution/value is unknown. Hold until the first completed-second BASE-net<=−2.5%
absorbing stop or entry+60s, capped by original HOT+60min/session close. Exit order
uses the first regular second open at/after its submission, also lag<=3s. Missing
or late entry/exit remains unresolved; no earlier-print settlement. Stop never
uses a second's low before completion. Costs are the unchanged LIGHT/BASE/STRESS
spread/slippage/minimum-cent scenarios; the stop is defined on BASE. Report actual
reference fill delays, cost-only stop breaches and all censoring reasons. These
are aggressive next-print scenario proxies, not proven NBBO/size fills.

## Causal WAIT label and two small value models

Fix baseline pi0: at each original observation ENTER iff current G>=its own
strict-past training prior and known round-trip cost drag<2.5%. Otherwise WAIT;
repeat at the next original observation until the inherited HOT+300s watch expiry.
Expiry with no entry is cash0, caused by a fixed timer, not a future best-state
choice. ENTER outcomes follow the fixed contract above.

E_i is the immediate ENTER BASE outcome. W_i is pi0's realized BASE outcome when
starting at the NEXT original observation, or cash at watch expiry. Future G
values are used only to simulate pi0's future causal actions when constructing
supervised labels. Censored E or selected future pi0 ENTER makes that value NaN.
This is Q^pi0_WAIT, not optimal waiting or a Bellman-optimal Q value. Learned policy
may differ from pi0; report that limitation rather than claiming consistency.

Fit independent linear ridge Q_E and Q_W, both on exactly the common finite E/W
rows. Inputs: pinned current G logit;7 pinned R328 S numeric gate-conditioned
coordinates; shared gate; known_cost_drag_pct/2.5 (10 inputs). No interactions,
new EMA, sequence model or threshold/horizon search. Standardized current columns
and G remain frozen; other inputs have fixed scales. Unpenalized intercept,
L2 alpha10 for slopes; solve deterministic weighted normal equations. Original
weight1/(number of ORIGINAL observed clocks in the episode), retained on finite
targets without renormalizing censored rows. No clipping/winsorizing returns.

Full fit only May6–8. May7 diagnostics fit May6; May8 fit May6–7; May6 has no
preceding value fit and stays unsupported/NaN. Chronology uses corresponding
preceding-scope R328 coordinates for fitting/scoring, not full-fit preprocessing.
Unsupported empty fits remain unsupported. Persist coefficients, input/target
support, condition numbers, normal-equation residuals and all full/fold provenance.

## Policies and independent episode replay

F: ENTER iff Q_E>max(0,Q_W) and causal cost drag<2.5%; otherwise WAIT and reevaluate.
E: same Q_E and cost veto, ENTER iff Q_E>0; no waiting-value comparison.
G: fixed pi0 recurrent control. G0: pi0 decision at original first clock only;
if rejected, remain cash. CASH: never enter. One filled/attempted entry per original
episode in this first experiment; HOLD/EXIT stays the fixed downstream contract.
Do not inspect future fill availability to admit an order. Unsupported value
predictions are explicit WAIT with an unsupported flag, not fabricated estimates.
Keep every episode, including no-observation cash and unknown trades.

## Shared capital diagnostic

Replay each policy's current decisions chronologically with at most2 simultaneous
positions,10% initial-day capital per order, and no simultaneous same-ticker
positions. Rank simultaneous F candidates by Q_E−max(0,Q_W), E by Q_E, G/G0 by
G−prior; ties use fixed identity keys. Capital unavailable means WAIT, with new
observations eligible for reassessment. Release capital only after an observed
exit fill. Unknown entry/exit blocks its reserved slot through the session and
makes that day's account outcome unknown. Do not rank by realized return.

Report daily account P&L, reserved exposure, skipped decisions, cash, opportunity
capture, severe losses and consistency. Aggregate account return is unknown if
any day is unknown. Closure-ledger drawdown is descriptive and cannot stand in
for intratrade marked equity drawdown, NBBO validation or a deployable allocator.
Daily initial capital reset is a controlled replay; an unresolved day invalidates
the claimed complete multiday account trajectory. No learned allocation/ADD/REDUCE
or HOLD/EXIT promotion is implied by this first bridge.

## Fixed evaluation and18 checks

Report full-known candidate mean BASE/STRESS return including cash; when any
episode is unknown, that full-population value is NaN and resolved-only means are
explicitly conditional. Report resolution, entries, trade returns, losses<=−2.5%,
per-date consistency and fixed-contract positive-opportunity capture (a future
diagnostic only, never an input). Paired F−E/F−G BASE intervals use the same common
resolved episodes, original equal-day/identity weights and1,000 fixed day and
ticker-day cluster multiplicities, seeds20266450/51. No refits; omission of unknown
outcomes is a stated limitation. Do not silently replace a gate metric with a
conditional one when its full-population value is unknown.

18 checks: paired training targets>=150 episodes; immediate positive BASE target
episodes>=30; positive/negative immediate targets each on>=3 dates; strict-prefix
fitting/no outcome inputs; immutable current/G path copies; F entries>=30;
F episode resolution/cash>=95%; all F shared-capital days resolved; full-known F
BASE mean>0; full-known F BASE mean>E and>G; full-known F STRESS mean>0;>=5/8 F
positive full-known daily candidate means; F closure-ledger drawdown<=E and G
when all three accounts complete; both-scheme paired F−E and F−G BASE lower CI>0
(4 checks). Preserve every failure. Underpowered support does not authorize a
changed horizon, cost, threshold or fit schedule; still report fixed diagnostic
fits unless inputs/linear solve are empty or numerically invalid.

## Verification and next decisions

Tests must cover next-observation WAIT labels vs future max, missing-fill censoring,
completed-close stops, deadline behavior, cost effects, label/future inference
invariance, strict chronology and same-target weights. Account tests must cover
simultaneous ranking, capacity WAIT/reassessment, delayed capital release,
same-ticker blocking and unknown outcomes. Atomic readable-footer parquet output.
Full tests/critical Ruff, one successful cached official replay (retain any failed
attempt), authenticated source/archives, JSON/coefficients/tables<=1e-9, exact
identities/missingness/action decisions/ranks. Retire workflow at closeout.

Failure must redefine the bottleneck: value-target/execution support, absolute
economic value, WAIT policy inconsistency or capital opportunity cost as evidence
indicates. Do not return automatically to memory variants. HOLD/EXIT, opportunity
cost of redeployment and learned allocation are separate later experiments.
Event R316/R316B remains independent. No main merge, model/policy promotion or live
trading, even if all reused-development checks pass.
