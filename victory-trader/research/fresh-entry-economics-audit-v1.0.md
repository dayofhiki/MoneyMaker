# Request 181 — fresh entry economics decomposition audit

## Status

Pre-registered after Request 180 failed its controller-improvement and economic
gates, before running Request 181.

Request 179 and 180 both show that the current fresh BUY population is already
deeply BASE-negative by the first executable post-entry minute. The old
short-HOLD comparator exits every reachable episode at minute 1 and still
realizes roughly -1.5% day-balanced BASE return. Therefore the next question is
upstream of HOLD/EXIT:

**Is the first-minute loss caused mainly by adverse price movement after entry,
or by modeled execution/fee drag on these cheap/illiquid names?**

No new market dates are opened and no policy is changed.

## Population

Reuse exactly the Request-178 BUY-gated POSITION episodes on:

- 2026-06-23
- 2026-06-24
- 2026-06-25
- 2026-06-26
- 2026-06-29

Use only episodes with exact minute-1 executable open references. For the
delayed-entry counterfactual, additionally require the exact minute-2 state/open.

## Audits

### Immediate entry, one-minute exit

For each episode:

- entry price = frozen BUY entry open;
- exit price = minute-1 executable open;
- gross return = raw open-to-open return;
- BASE return = existing modeled BASE round-trip return;
- execution drag = gross return - BASE return.

### One-minute WAIT counterfactual

No capital is committed at the original BUY time.

- delayed entry = minute-1 executable open;
- delayed exit = minute-2 executable open;
- gross delayed return = raw open-to-open return;
- BASE delayed return = the same frozen BASE execution model applied to the
  delayed entry and exit;
- delayed execution drag = delayed gross - delayed BASE.

This is an audit counterfactual, not an executable candidate policy.

## Required diagnostics

Report pooled and day-balanced:

- immediate gross, BASE and execution drag;
- delayed gross, BASE and execution drag;
- delayed-minus-immediate BASE difference;
- fraction of trades with positive gross but negative BASE;
- positive-rate and severe-loss-rate;
- by-day metrics;
- by-entry-price-bin metrics;
- exact minute-1 and minute-2 coverage;
- 10,000-sample trading-day bootstrap for delayed-minus-immediate BASE.

## Interpretation

Classify the dominant first-minute bottleneck descriptively:

- **execution-dominant** when mean execution drag exceeds the absolute adverse
  gross move;
- **price-move-dominant** when the absolute adverse gross move exceeds execution
  drag;
- **mixed** otherwise or when gross movement is positive.

A robust positive delayed-minus-immediate difference indicates entry timing is
worth a dedicated causal WAIT/ENTER model. Large execution drag indicates the
next intervention should strengthen executable-liquidity selection or fill
logic rather than continue tuning HOLD/EXIT.

Request 181 is diagnostic only and cannot promote a trading policy.
