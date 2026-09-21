# Expanded-history recurrent path policy v3.8 — pre-registration

## Status

Frozen after v3.7 request 95 passed all five bridge conditions in January,
February and March, and before implementing or inspecting any v3.8 trajectory
output.

Use only September 2025 through March 2026 development data. April 2026 and
later remain sealed.

v3.8 changes no predictive feature, lag, classifier, calibration rule,
opportunity threshold or HOLD threshold. It tests whether the already-supported
v3.7 one-step continuation signal remains economically useful when composed
into the single path that an executable position would actually traverse.

## Motivation fixed before results

v3.7 was the first continuation experiment to pass its frozen bridge in every
development month. On the frozen opportunity-gate stratum the day-balanced
one-step policy increments were +0.018844%, +0.025549%, and +0.033109% in
January-March, with trading-day bootstrap lower bounds +0.003151%,
+0.010545%, and +0.022094%.

Those rows are counterfactual diagnostics. A real position cannot visit states
after it has already exited. The next uncertainty is therefore trajectory
composition rather than another one-step model rewrite.

## Frozen upstream model and entry gate

For each monthly past-only fold, reproduce v3.7 exactly:

- exact v3.2 first-anchor panels from request 84;
- frozen 2025 state panels from request 54 and frozen 2026 state panels from
  request 8;
- the unchanged v3.3 opportunity model and strict
  P(opportunity) > 0.5 entry gate;
- the exact v3.7 HistGradientBoostingClassifier capacity and chronological
  Platt calibration;
- all ten v3.5 causal position-path variables;
- exact 1/2/3/5/8/13-minute deltas for every path variable;
- no gap compression;
- HOLD iff calibrated P(HOLD wins) > 0.5;
- no post-result phase, lag, duration or feature subset.

The conceptual entry remains the exact next-minute open after the frozen anchor.
Full round-trip entry and exit friction is applied in v3.8 because the unit of
evaluation is now a completed trade rather than an already-open one-step
decision.

## Frozen recurrent trajectory

For every evaluation anchor whose frozen opportunity probability is above 0.5:

1. attempt entry at the exact anchor+1-minute open;
2. if the exact entry open is unavailable or non-positive, record a missing
   entry and do not invent a fill;
3. after one completed holding minute, evaluate the frozen v3.7 classifier at
   each exact one-minute decision state;
4. if calibrated P(HOLD wins) > 0.5, remain in the position for one more minute;
5. otherwise exit at the exact currently observable decision open;
6. repeat only along states that the position actually reaches;
7. after 29 completed decision states, one final HOLD is allowed only to reach
   a hard 30-minute maximum holding time, after which exit is mandatory.

The 30-minute value is a safety/evaluation cap inherited from the v3.5-v3.7
geometry, not an optimized target horizon.

## Frozen missing-bar / halt behavior

No timestamp may be compressed.

If an expected decision-state bar is absent, or the exact decision open needed
for an EXIT is absent, stop model decisions for that position and mark it
`gap_exit_pending`. Exit at the first subsequent observed positive open for
that ticker-day. This delayed open is evaluation-only and may not be used to
change any prior action.

If the forced 30-minute exit open is absent, use the same first-subsequent-open
rule.

If no later open exists in the available panel, the position is unresolved.
Unresolved positions are never assigned zero return.

## Frozen execution scenarios

Apply the existing MoneyMaker execution model unchanged to both sides of every
completed trajectory:

- light: 10 bps half-spread + 10 bps slippage, minimum half-spread 0.5 cent;
- base: 25 bps half-spread + 25 bps slippage, minimum half-spread 1 cent;
- stress: 75 bps half-spread + 75 bps slippage, minimum half-spread 2 cents.

Use `modeled_buy_fill` at the entry reference and `modeled_sell_fill` at the
exit reference. Existing sell-fee handling remains unchanged.

Report raw gross return separately from friction-adjusted return.

## Frozen comparator

Use one primary timing comparator on exactly the same gated entries:

**always-HOLD-to-30m**: enter at the same exact next-minute open and remain in
the position until the 30-minute cap. If the exact 30-minute exit open is
missing, exit at the first subsequent observed positive open under the same
gap rule.

This comparator is fixed to isolate whether recurrent v3.7 exit timing adds
value relative to never acting on the continuation classifier. No fixed
5/10/15-minute comparator may be selected after outcomes.

## Required trade-level diagnostics

For each January-March evaluation fold report at minimum:

- gated entry attempts, valid entries, completed trades and unresolved trades;
- completion coverage;
- recurrent exit-reason counts;
- arithmetic gross/light/base/stress mean return;
- day-balanced base return;
- median, p05, positive-trade rate and severe-loss rate under BASE;
- mean, median, p10, p90 and maximum holding minutes;
- mean number of recurrent HOLD decisions;
- same trade-return summaries for always-HOLD-to-30m;
- matched recurrent-minus-comparator BASE return;
- 10,000-sample trading-day cluster bootstrap for recurrent day-balanced BASE
  return;
- 10,000-sample trading-day cluster bootstrap for the matched
  recurrent-minus-comparator BASE difference;
- strict fold provenance.

Descriptive breakdowns may not be used to alter the frozen policy.

## Frozen account replay

Replay recurrent and comparator trajectories independently for each
month/scenario with the existing audit conventions:

- synthetic initial cash = $10,000;
- fixed fractional order budget = $1,000;
- no leverage;
- an already-open same-ticker position blocks a new entry;
- insufficient free cash blocks an entry;
- exits at a timestamp are processed before entries at the same timestamp;
- simultaneous entries are processed by ticker as a deterministic audit
  tie-break, not as a claimed ranking edge;
- blocked attempts remain in the ledger;
- unresolved accepted positions retain committed capital and a last observable
  mark;
- report accepted, blocked, closed, unresolved, ending marked return and
  accounting completeness.

This is a historical bar-reference replay, not evidence of actual fills.

## Frozen promotion rule

v3.8 passes the development bridge only if **every** January, February and
March fold satisfies all of the following:

1. at least 100 completed recurrent trades;
2. recurrent completion coverage >= 90% of valid gated entries;
3. arithmetic BASE net mean > 0;
4. day-balanced BASE net mean > 0;
5. trading-day bootstrap 95% lower bound for recurrent BASE return > 0;
6. day-balanced recurrent BASE return strictly exceeds the matched
   always-HOLD-to-30m comparator;
7. trading-day bootstrap 95% lower bound for the matched recurrent-minus-
   comparator BASE difference > 0;
8. the recurrent BASE account replay has positive ending marked return and
   complete accounting.

All eight conditions must hold in all three months. A positive LIGHT result,
zero-trade abstention, or a pooled-only result cannot substitute for a failed
month.

If v3.8 passes, freeze this exact recurrent policy and preregister a genuinely
untouched validation month before accessing it.

If v3.8 fails, April remains sealed. Do not tune the 0.5 HOLD threshold, 30-minute
cap, lag grid, phase buckets or model capacity on January-March. Use the frozen
diagnostics to distinguish among three failure classes:

- one-step continuation edge does not compose into trajectories;
- entry economics plus full round-trip friction dominate the exit improvement;
- missing/halt/account constraints break otherwise positive isolated trades.

The next branch must address the diagnosed bottleneck rather than optimize the
failed v3.8 path against these same months.
