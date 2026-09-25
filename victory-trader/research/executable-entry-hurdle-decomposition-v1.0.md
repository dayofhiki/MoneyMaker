# Request 198 — executable-entry hurdle decomposition

## Why

Request 197 used one regression target: realized one-second risk-policy net
return, fitted only on closed trajectories. On chronological calibration every
frozen policy/threshold remained negative and closed coverage was only 70-88%.

That mixes several different failures:
1. can an entry actually fill within the five-second order lifetime?
2. after entry, does the path reach a causal terminal fill within the research
   window?
3. does the profitable branch occur before the stop branch?
4. conditional on a resolved trade, how large is the realized net return?

Request 198 separates these hurdles before another executable action policy is
built.

## Data and chronology

No new dates.

- train every head on Request-171 FIT only;
- evaluate every diagnostic on Request-171 chronological CALIBRATION only;
- Request-178 fresh development dates remain unopened by this request.

Use the same causal 1-second adapter, BASE friction, 1-second latency and
five-second entry expiry as Request 197.

Shadow candidate states remain minutes 1-10 with at least 31 regular-session
minutes remaining.

## Frozen rules

Evaluate both already-declared rules independently:

- tight: stop 3%, take 5% on 50%, proportional trail 5%;
- runner: stop 3%, take 10% on 50%, proportional trail 7%.

No new risk rule is introduced.

## Labels

For every candidate state:

### H1 — entry fillability
`entry_fillable = 1` if the five-second entry order obtains an eligible causal
one-second open, otherwise 0.

This label is rule-independent.

### H2 — terminal resolvability
Among filled entries:
`terminal_closed = 1` only if the risk replay obtains all required exit fills
and closes the position. Ambiguous and unresolved paths are 0.

### H3 — right-tail capture
Among filled entries:
`partial_take_reached = 1` if the causal policy actually fills its first partial
take before the episode terminates, otherwise 0.

This deliberately differs from a hindsight future-high label.

### H4 — positive economics
Among closed entries:
`positive_net = 1[realized BASE net return > 0]`.

Also retain realized net return as a regression diagnostic.

## Models

Use the same causal feature family available to Request 197.

For each rule:
- H1 HistGradientBoostingClassifier;
- H2 HistGradientBoostingClassifier;
- H3 HistGradientBoostingClassifier;
- H4 HistGradientBoostingClassifier;
- closed-return HistGradientBoostingRegressor.

H1 may be shared if the labels are identical across rules.

All fit weights are equal-day. No calibration offsets or thresholds are fit.

## Calibration diagnostics

Report for each head:
- support/prevalence;
- ROC AUC and average precision;
- per-day AUC when defined.

For closed return:
- Spearman;
- per-day Spearman when defined.

Build one frozen composite diagnostic score per rule:

`p_fill * p_closed_given_fill * p_positive_given_closed`

This is a ranking diagnostic only, not an executable policy.

Report P80/P90/P95 score groups on calibration:
- state count;
- episode count;
- entry-fill rate;
- closed coverage;
- actual positive-return rate among closed;
- actual day-balanced net return among closed;
- unresolved/ambiguous rates.

## Interpretation

This request cannot pass or promote a strategy.

Interpretation map:
- weak H1: historical 1-second aggregate executability itself is not learnable
  enough from current causal state features;
- strong H1 but weak H2: post-entry liquidity/terminal execution is the primary
  missing component;
- strong H1/H2 but weak H4: the remaining bottleneck is price direction / entry
  timing, not execution;
- strong individual heads and useful composite ranking: rebuild Request 197 as
  an explicit hurdle action model, then select its threshold on calibration.

Do not tune a new action threshold inside Request 198.
