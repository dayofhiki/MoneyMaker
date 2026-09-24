# Request 187 — fresh BUY horizon ceiling audit

## Status

Pre-registered after Request 186 failed and before running Request 187.

Requests 184-186 show that neither exact one-minute return prediction,
original-HOT three-minute opportunity prediction, nor one-minute WAIT followed
by re-observation produces a positive fresh entry policy.

Before changing another model, Request 187 asks a more basic question:

**Does the current fresh BUY population contain cost-positive executable exit
opportunity at all, and how quickly does that opportunity appear?**

No model is fitted, no policy is changed, and no new dates are opened.

## Population

Reuse exactly the Request-178 BUY-gated POSITION episodes on:

- 2026-06-23
- 2026-06-24
- 2026-06-25
- 2026-06-26
- 2026-06-29

## Horizon ceilings

For each BUY episode and each frozen horizon:

    1, 2, 3, 5, 10, 15, 30 minutes

compute the hindsight-best BASE round-trip return among exact executable exit
references available from entry through that horizon.

This is an oracle ceiling diagnostic. It is not an executable policy and must
never be reported as realized trading return.

For the 30-minute horizon, include the existing minute-30 safety-cap reference
when the minute-29 POSITION row contains an exact next-minute return.

## Required diagnostics

For each horizon report pooled and day-balanced:

- evaluable episode count and coverage;
- oracle BASE mean and median;
- positive-opportunity rate;
- p05;
- positive-opportunity days;
- per-day mean and positive rate;
- 10,000-sample trading-day bootstrap of the day-balanced oracle mean.

Also report the incremental gain in mean oracle value versus the previous
horizon.

## Interpretation contract

- If even the 30-minute oracle day-balanced mean is <= 0, the current fresh BUY
  population itself has no economic ceiling under the frozen BASE execution
  assumptions. Research must move upstream to opportunity/entry selection.
- If the 30-minute oracle is positive but short horizons are negative, the
  population contains opportunity but the current short-horizon entry/exit
  formulation is mismatched to the temporal shape. Research should focus on
  causal multi-minute path/value representation rather than one-minute action
  prediction.
- If a short horizon is already positive but models cannot select it, the main
  bottleneck is prediction/representation rather than population economics.

Request 187 is diagnostic-only and never promotion-eligible.
