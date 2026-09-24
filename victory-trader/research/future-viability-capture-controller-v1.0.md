# Request 192 — future-viability + capture recurrent controller

## Status

Pre-registered after Request 191 passed its frozen development bridge and
before implementing or running Request 192.

Request 191 established that causal POSITION-path information can identify
states from which a future exact BASE exit still covers modeled costs.
Request 189 established that causal POSITION information can rank relative
within-episode exit quality, but capture percentile alone held losing episodes
too long.

Request 192 composes those two signals into an honest one-minute-at-a-time
executable HOLD/EXIT controller.

No new dates are opened. The June 23/24/25/26/29 block remains development
data and cannot promote the model.

## Frozen upstream data

Historical fit/calibration:
- Request-171 BUY-gated rich-second POSITION rows;
- Request-173 market-regime context from the Request-163 historical scan.

Fresh development evaluation:
- Request-178 POSITION + scan shards on Jun 23/24/25/26/29.

Entry policy is unchanged.

## Frozen models

1. Exact Request-191 future cost-cover classifier and value regressor:
   - equal-day fit weighting;
   - classifier calibration gate = old calibration probability P75;
   - regressor residual offset from old calibration only.

2. Exact Request-189 capture-percentile model:
   - trained on old fit POSITION rows only;
   - frozen capture exit threshold = 0.70, the threshold already selected by
     Request 189 on old chronological calibration.

No threshold is selected or changed on June 23-29.

## Causal mark-to-market

At each completed POSITION state, compute a causal marked BASE return using:
- the known modeled entry fill from the actual entry open;
- the just-completed causal current close;
- the same BASE sell model and sell fee.

The next executable open remains outcome-only and is never used to decide.

## Frozen recurrent rule

At each minute from 1 through the inherited 30-minute research cap:

### If state is unscoreable
EXIT at the next executable open.

### If marked BASE return is positive
EXIT if either:
- predicted capture percentile >= 0.70; or
- Request-191 does not give high future-viability confidence; or
- predicted best future BASE return <= the current causal marked BASE return.

Otherwise HOLD one more minute.

### If marked BASE return is zero or negative
HOLD only when both:
- future cost-cover probability >= the frozen Request-191 calibration P75 gate;
- predicted best future BASE return > 0.

Otherwise EXIT and cut the position.

At minute 30 force EXIT using the same inherited safety-cap semantics.

## Comparators

Evaluate on the exact same fresh BUY episodes:
- minute-1 EXIT;
- HOLD to minute 30;
- Request-189 capture-only controller at threshold 0.70;
- Request-192 composed controller.

## Required economic diagnostics

For each policy:
- started/completed episodes and completion coverage;
- mean and day-balanced BASE return;
- median and p05 BASE return;
- positive-trade rate;
- severe-loss rate (<= -5%);
- mean/median/p90 hold time;
- day-cluster bootstrap 95% interval.

For Request 192 additionally:
- matched return difference versus minute-1 EXIT;
- matched return difference versus Request-189 capture-only;
- positive mean-return trading days.

## Reference account ledger

To make results intuitive, report a separate reference account view:

- starting balance: KRW 1,000,000;
- maximum concurrent positions: 5;
- each newly admitted trade receives up to 20% of current nominal equity
  (cash + cost basis of open positions), bounded by available cash;
- exits return allocated principal multiplied by (1 + realized BASE return);
- exits at the same timestamp are processed before new entries;
- episodes skipped only because all five slots are full are counted;
- unresolved trajectories are excluded from the balance claim and the ledger
  must report completed-episode coverage.

This account view is a transparent allocation illustration, not a new model
selection metric.

## Frozen development gate

Request 192 passes only if all hold:

1. start-state coverage >= 85%;
2. completion coverage >= 90%;
3. Request-192 day-balanced BASE return > 0;
4. Request-192 day-cluster bootstrap 95% lower bound > 0;
5. Request-192 mean BASE return is positive on >=4/5 days;
6. severe-loss rate is no worse than minute-1 EXIT;
7. matched day-balanced improvement over minute-1 EXIT > 0 with bootstrap
   95% lower bound > 0;
8. matched day-balanced improvement over Request-189 capture-only > 0;
9. reference-account ending balance > KRW 1,000,000.

A pass still does not promote the model because the fresh block has already
been opened repeatedly. It licenses a preregistered untouched-date validation.
