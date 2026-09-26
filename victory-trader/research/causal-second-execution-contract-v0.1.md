# Causal one-second execution contract v0.1

## Why this exists

The corrected Request 194 lost money on its June development evaluation even
though its minute-close risk rule reduced severe losses. Request 196A lowered
the fixed-horizon winner AUC to 0.613. Neither result supports live trading or
claims of a profitable entry edge. More fundamentally, Request 194's `-3% stop`
was only inspected at completed minute marks. This contract removes that
semantic mismatch before another executable entry experiment is evaluated.

`causal_second_execution.py` is a single-long-position reference simulator.
It is not yet connected to the shared portfolio ledger or attention runtime.
Historical one-second OHLC bars and synthetic friction cannot establish actual
broker fills.

## Frozen timing and ambiguity rules

1. A decision is known at `decision_t`. Entry uses the first active one-second
   bar open at or after `decision_t + latency_ms`, expiring after five seconds
   by default. There is no entry on a stale later bar.
2. A second bar stamped `t` reveals its high/low/close only at `t + 1s`.
   A barrier touching that bar submits an exit no earlier than `t + 1s`; the
   first eligible later active open, after configured latency, is the synthetic
   reference fill. The barrier price is never granted as the fill price.
3. Stop and take reached in the same second before a partial take yield an
   ambiguous, unscored trajectory. A newly raised trailing high and its
   trailing breach in the same second are also ambiguous if the old trailing
   level was not already breached. Missing seconds do not themselves trigger
   actions.
4. An explicitly halted second cannot fill. Pending orders remain pending
   through a halt and may fill at a worse resume open. A missing exit open
   leaves the trajectory unresolved. A stop percentage is a trigger and sizing
   reference, never a maximum-loss guarantee.
5. The take sells a fixed fraction. Trailing is a proportional decline from
   the post-partial price high, not a subtraction of return percentage points.
   The stop remains active on the remaining position. The time cap is based on
   elapsed time from entry and also requires a later executable open.
6. Whole-share sizing uses account risk at a modeled stop, capped by cash,
   maximum capital allocation and an optional liquidity share cap. Scenario
   costs are synthetic and must be stress tested.

## Next research experiment

Before reading outcomes, adapt the saved causal second paths to this contract,
add explicit halt intervals and a single mark-to-market account ledger, and
check timestamp/coverage consistency against the existing minute and scan
records. The Request 171 FIT block should train an entry model. The later
chronological calibration block should freeze its ENTER/WAIT threshold and a
small declared risk-rule family. June 23/24/25/26/29 has already been opened
for development and can diagnose implementation only; it is not promotion
evidence. Keep later untouched periods sealed until the policy, cost scenarios,
latencies, missing-outcome handling and success gates are committed.

The entry target should be path ordered: reward before stop, or the net result
of a frozen causal risk replay. Compare actual admitted trades with WAIT and
simple baselines using cost-adjusted return and mark-to-market account equity,
not state-level AUC alone. Report day/episode clustered uncertainty, open-order
and unresolved rates, halt-gap losses, turnover and position overlap. Run at
least 0/1/2/5-second latency and light/base/stress friction sensitivity. An
unresolved path must never be dropped silently from a profitability claim.

Promotion requires positive, stable *net* account economics on a genuinely
untouched chronological period, adequate coverage and no accounting or causal
integrity failures. The present implementation does not meet that gate.

