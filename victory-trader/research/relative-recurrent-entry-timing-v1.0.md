# Request 207 — relative recurrent entry timing

## Status

Pre-registered after Request 206 failed its development gate.

Request 206 successfully exercised repeated WAIT states but almost never entered:
223 episodes produced 590 WAIT actions and only two recurrent entries. The
absolute ENTER>0 / WAIT>0 value boundary therefore behaved as a calibration
bottleneck rather than a useful timing controller.

No new dates are opened in Request 207.

## Question

For an already admitted candidate after the first WAIT, can the model answer the
simpler relative question:

> Is entering now better than waiting exactly one more minute?

The answer is recomputed at every exact minute checkpoint.

## Target

At minute m:

- current value = realized BASE return from entering at m and exiting exactly
  three minutes later;
- next value = the same return definition from entering at m+1;
- relative advantage = current value - next value.

Future outcomes are labels only. Inputs remain the same causal market-state
features from Request 206 and exclude position P&L/path columns.

## Runtime

At minutes 1..4:

- ENTER when predicted relative advantage >= 0;
- otherwise WAIT one minute and re-observe.

At minute 5, ENTER if an executable three-minute evaluation exists.

This intentionally removes ABSTAIN and absolute expected-return gating. The
experiment isolates timing quality only. Upstream admission and later
HOLD/EXIT remain separate problems.

## Comparator

Always enter at minute 1.

Evaluate both on matched episodes using the same fixed three-minute BASE exit.

## Frozen development gate

Pass only if:

- minute-1 state coverage >= 80%;
- recurrent policy has at least 100 trades;
- WAIT is used in >=10% of episodes;
- matched day-balanced recurrent-minus-minute1 return > 0;
- matched day-cluster bootstrap 95% lower bound > 0;
- recurrent timing improves at least 4/5 days;
- severe-loss rate is no worse than minute-1 entry.

Absolute profitability is reported but is not a gate here. This request asks
only whether repeated relative timing adds value.
