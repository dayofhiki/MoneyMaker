# Request 208 — pullback / turn transition entry representation

## Status

Pre-registered after Request 207 completed and failed its development gate.

Request 207 proved the recurrent machinery is active: 200 WAIT actions were
taken across 223 episodes and entries were distributed from minute 1 through
minute 5. Relative timing modestly improved day-balanced BASE return by about
+0.020 percentage points and reduced severe-loss rate, but the direct
ENTER-now versus WAIT-one-minute target was not observable from the snapshot
representation (fresh Spearman -0.0166).

No new dates are opened.

## Hypothesis

A trader does not identify a pullback and turn from one isolated snapshot. The
decision depends on how state has changed since the previous observation and
since the watch episode began.

Add only causal transition geometry computed inside the already-observed WATCH
episode:

- one-minute price change and its acceleration;
- return from the first WAIT state;
- drawdown from the post-HOT observed price peak;
- rebound from the post-HOT observed price trough;
- one-minute attention-score and attention-rank changes;
- attention-score drawdown from its post-HOT peak;
- one-minute changes in log volume and log transactions;
- one-minute change in active-second count;
- changes in 5s/10s momentum, close-vs-VWAP and short burst measures.

No future, old-position P&L, old entry price, or oracle columns are inputs.

## Target and runtime

Keep Request 207 unchanged:

- target = BASE 3m return entering now minus BASE 3m return entering exactly
  one minute later;
- minutes 1..4: ENTER when predicted relative advantage >= 0, else WAIT;
- minute 5: enter when evaluable;
- comparator = always enter at minute 1.

Only the causal representation changes.

## Frozen development gate

Pass only if:

- minute-1 state coverage >= 80%;
- recurrent trades >= 100;
- WAIT used in >=10% of episodes;
- fresh relative-advantage Spearman >= +0.05;
- matched day-balanced recurrent-minus-minute1 return > 0;
- matched day-cluster bootstrap 95% lower bound > 0;
- recurrent timing improves at least 4/5 days;
- severe-loss rate is no worse than minute-1 entry.

This is still development-only and promotion-ineligible. A pass would justify
carrying transition-state timing into an integrated entry/position controller.
