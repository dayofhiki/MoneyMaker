# One-second path to account bridge v0.2

## Scope

This extends the v0.1 single-position execution contract with two research
components: a strict adapter for Massive-style `t,o,h,l,c` one-second aggregates
and a chronological, marked multi-position account replay. It is implementation
work on already-open research data, not a new profitable-strategy result.

`adapt_second_bars` accepts explicit regular-session bounds and optional
halt/resume intervals. It rejects duplicate timestamps instead of silently
keeping the last conflicting second. Any second overlapping a supplied halt is
unusable for a fill or trigger, including a partial boundary second. An empty
halt list means only that no halt annotations were supplied; it cannot prove
there were no halts. The adapter does not invent zero-volume seconds.

`replay_portfolio` consumes frozen `EntryAttempt` decisions. Each attempt must
carry a price observable **at its decision time** and an exclusive session-close
timestamp. The decision price is for sizing only; the later open is the synthetic
entry reference. Future bars are used offline to schedule potential evaluation
events, never to choose account admission or share count. At one timestamp,
expired orders and completed-second marks are processed before decisions; same-
timestamp exit/entry opens are processed afterward. Thus a new decision cannot
spend proceeds from an exit open it has not yet observed.

Orders reserve decision-time cash. If the later open is too expensive for the
already-frozen integer share count, the entry is rejected and never resized
with hindsight. Same-ticker overlap and portfolio-capacity violations are
blocked. A partial take sells a whole-share quantity and retains at least one
share. Missing entries release reservations; missing/ambiguous exits keep the
position and its capital unresolved. The equity curve marks remaining shares at
the last observed aggregate close and reports observed-mark drawdown, not a
broker-liquidation drawdown.

## Verification and remaining limits

Synthetic path tests cover gap losses, partial cash release, same-ticker
blocking, reservation expiry, entry gaps, session boundaries, ambiguity, and
future-price independence of simultaneous admission. These tests establish
accounting/timing behavior; they say nothing about positive expected return.

The next development-only diagnostic should adapt raw second bars for the
already-open June 23/24/25/26/29 HOT episodes, attach the official halt intervals
where coverage exists, and reconcile every attempted trade against the minute
scan's causal decision time. Report: adapter rejection count, missing second
coverage, halt coverage, entry expiry, ambiguous path rate, unresolved session
positions, cash/capacity blocks, and BASE/light/stress equity curves at 0/1/2/5s
latency. Do not drop unresolved trades or promote from these seen dates.

Before a fresh evaluation, the policy needs one frozen rule for session-close
liquidation versus unresolved overnight capital, a point-in-time entry feature
set, FIT-only training, separate chronological calibration, and explicit net
account success gates. Actual trade/quote or paper data remains necessary for
execution-cost and latency validation.

