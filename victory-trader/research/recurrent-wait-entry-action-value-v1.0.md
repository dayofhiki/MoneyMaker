# Request 206 — recurrent WAIT-state entry action value

## Status

Pre-registered after the project moved from one-shot entry timing toward a
continuous trader. This is a development-only structural diagnostic and opens
no new dates.

## Question

After the trader has chosen WAIT once, can it keep re-observing the market and
decide ENTER / WAIT / ABSTAIN from changing causal state, instead of being
forced to enter exactly one minute later?

## Population

Reuse the frozen Request-171 fit/calibration POSITION artifacts and the already
opened Request-178 June 23/24/25/26/29 block.

POSITION artifacts are used only as a source of market snapshots. The policy
inputs exclude entry price, running P&L, drawdown/recovery from the old
position, and future execution references.

Decision checkpoints are exact minute 1 through minute 5 after the original
HOT event.

## Action values

At each checkpoint:

- ENTER value target = BASE return from entering at the next causally
  executable open and exiting exactly three minutes later.
- WAIT value target = the recursively defined non-negative value available at
  the next checkpoint.
- ABSTAIN value = 0.

Two regressors estimate ENTER and WAIT value. Runtime chooses:

1. ENTER if predicted ENTER > 0 and >= predicted WAIT;
2. WAIT if predicted WAIT > 0 and another checkpoint exists;
3. otherwise ABSTAIN.

No class balancing, percentile threshold, or fresh-outcome tuning is allowed.

## Why this formulation

Historical BUY/WAIT/SKIP classification collapsed into the dominant SKIP class.
Request 186 showed that WAIT must mean real re-observation, not a delayed order.
Request 206 therefore treats timing as a sequential value problem.

## Comparators

- always ENTER at minute 1;
- one-shot minute-1 ENTER iff predicted ENTER value > 0;
- recurrent action-value ENTER/WAIT/ABSTAIN.

All realized policy returns use a fixed three-minute exit in this bridge so the
entry controller can be isolated from the still-unresolved HOLD/EXIT problem.

## Frozen development gate

The recurrent policy passes this structural bridge only if all hold:

- minute-1 state coverage >= 80%;
- at least 30 recurrent trades;
- WAIT is actually used in >=10% of episodes;
- recurrent day-balanced BASE mean > 0;
- recurrent day-balanced BASE mean > one-shot positive-entry comparator;
- severe-loss rate is no worse than the one-shot comparator;
- trading-day bootstrap 95% lower bound > 0;
- positive recurrent mean on at least 4/5 days.

A pass does not promote a live policy. It justifies combining the recurrent
entry controller with a causal recurrent HOLD/EXIT controller before opening a
new untouched block.
