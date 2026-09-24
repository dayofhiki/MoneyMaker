# Request 188 — relative remaining-value decay controller

## Status

Pre-registered after Request 187 established a robust positive 30-minute
hindsight ceiling on the fresh Request-178 BUY population and before running
Request 188.

Request 187 changes the diagnosis materially:

- 1-5 minute oracle ceilings remain BASE-negative;
- the first positive day-balanced ceiling appears at 10 minutes;
- the 30-minute oracle ceiling is +0.8572% day-balanced with a +0.3449%
  bootstrap lower bound and positive mean on all five fresh days.

Therefore the BUY population contains economic opportunity, but that opportunity
often arrives later than the one-to-three-minute action targets tested in
Requests 179 and 184-186.

Request 178 also showed that the consensus remaining-value score retains fresh
ranking while its absolute zero boundary does not transfer. Request 188 tests a
boundary-free recurrent controller that uses only the *direction of change* in
that score within the same open position.

No new dates are opened.

## Frozen input

Reuse the exact Request-178 scored POSITION rows and consensus hurdle-EV score:

    consensus_hurdle_ev_pct

No model is retrained and no score offset or threshold is changed.

## Relative trend signal

For consecutive reached POSITION states in the same episode:

    score_delta_t =
        consensus_score_t - consensus_score_(t-1)

The absolute score level is ignored.

Diagnostic targets:
- change in realized excess remaining-option value;
- current one-minute HOLD advantage;
- current remaining-option value.

The question is whether a falling score means the remaining opportunity is
actually decaying.

## Executable development controller

At each Request-178 BUY episode:

1. require the minute-1 POSITION state;
2. HOLD from minute 1 to minute 2 to obtain one within-position score change;
3. from minute 2 onward, compare current consensus score with the immediately
   previous reached score;
4. EXIT now at the first state where current score <= previous score;
5. otherwise HOLD exactly one more minute and re-evaluate;
6. if the next causal state is missing after a HOLD, exit at the already-defined
   next-minute executable reference when available;
7. after minute 29, a HOLD forces the minute-30 safety-cap exit.

No absolute EV threshold, percentile, drawdown amount, patience count or
fresh-tuned parameter is allowed.

## Comparator

Use the current short controller on the same episodes:

- EXIT at minute 1.

This is the actual behavior observed from the frozen Request-142 short-HOLD
classifier on the Request-178 block.

## Required diagnostics

Pairwise trend diagnostics:
- score-delta vs actual excess-value-delta Spearman;
- mean current one-minute HOLD advantage for rising versus non-rising score;
- mean current remaining-option value for rising versus non-rising score;
- daily versions of those comparisons.

Trajectory diagnostics:
- start-state and completion coverage;
- BASE mean and day-balanced mean;
- median/p90 holding time;
- positive-trade and severe-loss rates;
- 10,000-sample trading-day bootstrap;
- matched difference versus minute-1 EXIT and bootstrap;
- exit reasons.

## Frozen development gate

The relative controller is worth further work only if all hold:

1. start-state coverage >= 85%;
2. completion coverage >= 90%;
3. score-delta vs actual excess-value-delta Spearman > 0;
4. non-rising-score states have lower mean remaining-option value than
   rising-score states;
5. candidate day-balanced BASE return > 0;
6. candidate day-bootstrap 95% lower bound > 0;
7. matched candidate-minus-minute1 day-balanced difference > 0;
8. matched-difference bootstrap 95% lower bound > 0;
9. severe-loss rate is no worse than minute-1 EXIT.

This is development-only because June 23/24/25/26/29 are already opened.
A pass would freeze the relative controller for the next untouched block.
