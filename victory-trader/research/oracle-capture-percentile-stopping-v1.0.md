# Request 189 — oracle-capture percentile stopping

## Status

Pre-registered after Request 188 rejected the within-position score-decay rule
and before implementing or running Request 189.

The evidence now supports a specific temporal mismatch:

- Request 187: the fresh BUY population has a robust +0.8572% day-balanced
  30-minute hindsight ceiling, with a +0.3449% trading-day bootstrap lower
  bound and positive mean on all five days;
- Requests 179/184-186: one-to-three-minute action targets are too noisy or too
  economically early;
- Request 188: the Request-178 remaining-value score has weak positive temporal
  delta correlation but its local slope is not an exit signal.

Request 189 therefore learns exit *quality along the whole 30-minute path*
directly, using a bounded relative target rather than an absolute return or EV
boundary.

No new dates are opened.

## Target

For each historical BUY episode, collect every exact executable BASE exit
return from minutes 1 through 30.

For each causal POSITION row at minute m, define:

    capture_percentile_m =
        fraction of that episode's executable exit returns
        that are <= the current minute-m exit return

Properties:
- bounded in (0, 1];
- 1.0 means the current exit is the best observed executable exit in that
  episode;
- invariant to absolute return level and much less sensitive to cross-regime
  return offsets;
- uses future information only as a training label.

The live policy never observes this target.

## Causal state

Use the existing Request-171 causal POSITION representation:

- Request-166 minute/path features;
- original one-second aggregate features;
- Request-171 rich one-second aggregate features.

Execution-reference prices and future labels remain excluded from model inputs.

## Model

Fit one HistGradientBoostingRegressor on the historical fit partition:

- squared-error loss;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- minimum 75 samples per leaf;
- L2 regularization 2.0;
- equal total sample weight per trading day;
- seed 20261091.

Clip predictions to [0, 1].

No fresh calibration offset is allowed.

## Threshold selection

The only policy threshold is selected on the old chronological calibration
partition, before evaluating Request-178 dates.

Frozen candidate grid:

    0.70, 0.80, 0.90

For each candidate, run the full recurrent calibration trajectory:

- EXIT when predicted capture percentile >= threshold;
- otherwise HOLD one minute and re-evaluate;
- force exit at 30 minutes;
- preserve existing missing-state fallback.

Among candidates with completion coverage >=90%, select the threshold with the
highest calibration day-balanced BASE return. Ties choose the lower threshold
for earlier risk release.

If no candidate reaches 90% completion, the experiment is an implementation
failure.

No threshold may be changed after fresh results.

## Fresh executable policy

On each Request-178 BUY episode:

1. score minute 1;
2. EXIT now when predicted capture percentile >= the frozen calibration
   threshold;
3. otherwise HOLD exactly one minute;
4. repeat on each reached state;
5. force exit at minute 30.

Comparators:
- minute-1 EXIT;
- always HOLD to minute 30.

## Required diagnostics

- calibration threshold table and selected threshold;
- fresh capture-percentile Spearman;
- start and completion coverage;
- BASE mean/day-balanced/median/p05;
- positive and severe-loss rates;
- mean/median/p90 hold;
- exit reasons;
- trading-day bootstrap;
- matched difference versus minute-1 EXIT and bootstrap;
- matched difference versus hold-to-30m.

## Frozen fresh development gate

Pass only if all hold:

1. start-state coverage >=85%;
2. completion coverage >=90%;
3. fresh capture-percentile Spearman >= +0.10;
4. candidate day-balanced BASE return >0;
5. candidate day-bootstrap 95% lower bound >0;
6. matched candidate-minus-minute1 day-balanced difference >0;
7. matched difference bootstrap lower bound >0;
8. candidate severe-loss rate <= minute-1 EXIT severe-loss rate.

This is development-only because June 23/24/25/26/29 are already opened.
A pass would freeze the exact model and calibration-selected threshold for an
untouched forward block.
