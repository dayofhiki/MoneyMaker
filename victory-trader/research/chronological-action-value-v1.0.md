# Request243 — chronological causal policy continuation

## Why this experiment

The supplied conversation stopped at Request238, but the live repository had
already completed 239–242. Request242 (run 36255278684) found, in 43 admitted
June23–29 episodes, a first-event mean of -1.2695%, hindsight best-event3 mean
+0.7457%, and hindsight best-multi-event mean +0.6231%. Those oracle means include
cash on nonpositive opportunities; they are not deployable returns. Their
state-level rank correlation was 0.9052. This does not prove a target mismatch
or a profitable learnable edge. It does motivate testing causal action learning.

Request243 preserves those branches and isolates a new controller on archived
Request171/178 positions. It is explicitly not a direct performance comparison
with the 43 availability-admitted episodes: all archived BUY-gated episodes are
used, and every baseline shares the same population.

## What the targets mean

- ENTER: net proceeds from entering now, then following a frozen teacher's
  causal HOLD/EXIT decisions, including one full round-trip BASE cost.
- WAIT: realized value of the teacher's future flat policy after one observed
  state, including the possibility of never entering and staying in cash.
- HOLD-minus-EXIT: future teacher liquidation proceeds versus liquidation now.
  The entry cost is sunk and is not paid again at every HOLD decision.
- ABSTAIN: zero, with no position opened. EXIT is the zero reference for the
  HOLD advantage, not a fabricated zero realized trade return.

No target takes a maximum over realized future returns. Actual continuation
outcomes can be negative. Future outcomes are training labels; only observed
causal features determine the teacher's actions.

An initial one-event baseline produces early-fit targets. A teacher is learned
on Apr30, May1 and May4. It is frozen, then acts on May5–8 to generate the final
student targets. The student is fitted only on those later days. This is one
policy-improvement pass, not convergence or a claim of optimal action values.
Training uses an explicit 48-column prior causal allowlist plus causal path and
clock features. Future fills, labels, original position P&L, identity and future
availability never enter the learner.

## Execution and missingness

The source pipeline defines `state_t` as completed-minute availability time and
maps next-minute opens to that same time. Therefore the recorded
`exit_reference_open` is an execution label, not the completed bar's own open.
No extra minute of latency is silently added.

Flat decisions occur at the first five observed states within five clock minutes.
Once entered, HOLD/EXIT is reconsidered at each observed state. Minute 29 is a
precommitted liquidation deadline because the archived positions stop at 29.
A missing deadline is not replaced with the hindsight last observed row. Missing
fill references or liquidation prices stay unresolved and invalidate that day's
primary mean and the research gate. Missing targets are excluded from training
with counts reported; the censoring risk remains a limitation.

## Locked evaluation

The JSON request is the predeclared contract. Eight calibration dates are purely
validation data: no fitted offsets, score percentiles, model selection, or repeated
threshold search. June23–29 is already-opened development data and is loaded only
if calibration passes. No new market dates or API collection are requested.

The student must beat cash, the earlier teacher, a one-shot no-WAIT ablation,
an immediate-exit no-HOLD ablation, and the initial first-event one-step baseline
on the same equal-episode-notional daily proxy. It must have at least 20 resolved
trades, no unresolved trades, positive average trade return, a positive daily
bootstrap lower bound, positive days in a strict majority, and at least five
WAIT and HOLD actions. Stress returns are reported as a separate diagnostic.

These are not compounded portfolio returns: simultaneous positions, shared cash,
capacity, repeated entry and universe-wide coverage remain unmodeled. The BUY-gated
source's selection bias is inherited. Even a pass cannot promote a live trader.
