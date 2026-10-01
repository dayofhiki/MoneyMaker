# R298 design — clock-based orders, pending liquidation, honest path coverage

Status: designed, NOT dispatched. No research-request.json change.
Source dependencies: R297 run 36798494177 and R294 run 36721315072.
Data: existing May5-8 only. June HOLD validation and final July-August sealed.

## Question

How much of R297's failure comes from sparse-print execution/label missingness,
and how much remains a genuinely weak HOLD signal after those defects are fixed?
This is an execution-integrity experiment, not a parameter optimization round.

## Frozen components

Keep the same 86 entries and 97 candidate denominator, R296 utility/turn gates,
disabled override, 20 HOLD features, 60s/300s horizons, gradient-boosting settings,
leave-one-day-out folds, positive-advantage HOLD rule, -2.5% modeled-net hard stop,
2% old trailing comparator and LIGHT/BASE/STRESS scenario definitions.
No entry rejection, stop widening, extra features, threshold grid or new dates.

## A. Clock and order state machine

Maintain independent market-data events and scheduled timer events.
States: HOLD -> EXIT_SUBMITTED -> FILLED, or EXIT_SUBMITTED -> RIGHT_CENSORED.

- Fixed holds submit liquidation at actual entry_t + 60/300/600 seconds even
  when there is no trade print at that time. Old trailing submits at entry+600s
  if no prior stop; research terminal timer is min(HOT+60m, regular-session end).
- Model, trailing and hard-stop decisions use only completed observed seconds.
  Timer signals require no new price feature. While exit is pending, retain the
  position and original submission time; do not let later model scores cancel it.
- Fill reference is the first observed second OPEN at/after order submission,
  inside the regular session. Pay its full price gap and modeled friction.
  A later print may be after the observation cap; report the extra exposure.
- Record signal time, submission time, fill time, delay, last-observed-price age,
  observed jump, delay bucket and pending duration. Never fabricate a zero delay.
- A missing session-end fill is right-censored. Never use an after-hours open or
  retrospectively substitute the last earlier close as an executable fill.
- Keep the original <=3s flag as an execution-quality diagnostic, not permission
  to discard all other positions. Results must show the full denominator and
  right-censored count alongside the <=3s diagnostic subset.
- Next-print execution is an explicit historical reference scenario, not proof
  of order-book liquidity, quote availability or guaranteed broker fills.

The completed-state artifact cannot reconstruct the entry-bar start and every
terminal/pending fill safely in all cases. Reuse cached raw daily second bars;
fetch only missing cached ticker-days, on existing May dates, and log provenance.
A missing print must not be labeled a trading halt without independent evidence.

## B. Separate opportunity measurement from endpoint execution

- Observed-window MFE/MAE uses bars within actual entry to HOT+60m/session cap.
  MFE uses observed open prices and the entry reference, not favorable bar highs.
- Exact cap liquidation is not required merely to describe observed-window
  opportunities. Label these excursions as observed-window bounds; show active
  seconds, last-print age, gaps, endpoint coverage and sparse-path strata.
- For filled positions, measure post-exit observed upside only before the cap.
  Do not make absent observations mean zero missed upside. For pending exits
  extending past the cap, report the exposure separately.
- Net-stop decomposition: report unchanged-price friction breaches and
  price-deterioration breaches separately. Do not change the -2.5% stop in R298.

## C. Attribution sequence, with no model selection

1. Reproduce R297 exactly from saved states/scores; verify all resolved returns,
   unresolved reasons and STFS anomaly before any correction.
2. Replay the same saved dynamic scores using corrected pending execution;
   replay fixed timers at exact clock deadlines. Isolate simulator effects.
3. Rebuild the SAME 60s/300s continuation targets with pending execution rules.
   The future liquidation order is submitted at min(decision+horizon, cap),
   and its next regular-session open is the label; EXIT-now uses the same rules.
   Missing same-session outcomes remain NaN, with explicit coverage reporting.
4. Refit unchanged HOLD models on other development days only and replay them
   with the corrected engine. Compare to step 2 to isolate label/refit effects.
   No best-future-peak target and no learned threshold tuning.

Use observed-open targets as scenario labels, with target fill delays and an
explicit long-delay sensitivity. Do not portray an hour-delayed label as the
same thing as an executable sixty-second return.

## Reporting

For each policy and each stage show 86 submitted positions, filled/censored
counts, full candidate denominator, returns by cost scenario, daily means,
median, CVaR5, stop decomposition, time-to-fill, and gross/net opportunity capture.
Do not publish a full candidate P&L when any open position has unknown liquidation.
Use matched comparisons with exact sample counts; show unmatched/censored flows.
Report the largest trade, contribution concentration, and leave-largest-out mean.
Attribution bootstrap uses trading-day and ticker-day clusters; four days are a
small exploratory sample, not an independent profitability certification.

## Integrity acceptance criteria

- R297 reproduction agrees within 1e-9 on resolved returns and decision reasons.
- Every fixed timer fires at the intended wall-clock timestamp.
- All fills occur after submission and before official session close.
- All 86 entries reconcile to filled or explicitly right-censored, with no drops.
- Pending exits cannot be overwritten, reset, canceled or treated as cash.
- Future-bar mutation cannot change earlier features, scores or signal times.
- Quote-free liquidation assumptions and late-fill labels are visibly identified.
- Target coverage, delay distribution and terminal observation coverage are
  reported. Do not lower the R297 95% resolution expectation to obtain a pass.

Tests must include long silent gaps crossing 5m/10m timers, a gap-through stop,
next print beyond the research cap but before session close, no session-end fill,
after-hours-only next print, holiday/early-close bounds, future mutation,
pending-order persistence and strict JSON serialization.

## Decision boundary

If execution integrity remains unresolved, stop policy promotion and specify
what trade/quote data is needed. If integrity passes but fixed-horizon HOLD
signals remain nonpositive, design R299 policy-consistent continuation targets
on separate development data; do not tune June. A positive May mean alone does
not unseal June: the complete HOLD policy and evaluation must first be frozen.
