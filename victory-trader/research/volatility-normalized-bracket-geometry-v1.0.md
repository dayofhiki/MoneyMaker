# Request 206 — volatility-normalized bracket geometry bridge

## Purpose

Requests 199-205 repeatedly found the same pattern: several causal context
families improve relative runner ranking, but the selected tail remains
economically negative under the fixed -3% stop geometry.

Before adding any more information sources, Request 206 isolates the risk
geometry itself.

Hypothesis:

**a fixed -3% stop is not the same amount of risk across small-cap runners with
very different local volatility. A volatility-normalized 1R stop may preserve
valid runner paths that the fixed stop labels as immediate failures.**

This request has no predictive model and does not open fresh dates.

## Evaluation data

Use Request-171 chronological CALIBRATION only.

Request-178 remains sealed.

Each HOT episode contributes at most one attempted state:
- inspect causal shadow states from minute 1 through 10 in chronological order;
- require a valid volatility estimate and a causally executable entry;
- choose the earliest state satisfying both;
- if an earlier state has no fill, continue observing;
- once an entry is admitted, stop searching that episode.

All four bracket policies are replayed from the exact same admitted state and
entry fill.

## Causal volatility unit

At decision timestamp t use only one-second aggregates strictly before t.

Lookback:
- [t - 5 minutes, t).

Bucket into completed wall-clock one-minute intervals. For each active minute:

    minute_range_pct = (minute_high / minute_low - 1) * 100

Require at least 3 active minute buckets.

Define:

    raw_R = median(last up to 5 minute_range_pct values)
    R_pct = clip(raw_R, 3%, 8%)

The 3% floor preserves the previous minimum risk allowance. The 8% cap prevents
the risk budget from expanding without bound in extreme runners.

Do not tune the window, minimum bucket count, statistic, floor, cap, or R
multiples after seeing calibration outcomes.

## Frozen bracket policies

1. fixed_tight:
   - stop = -3%;
   - full take = +5%.

2. fixed_runner:
   - stop = -3%;
   - full take = +10%.

3. adaptive_2R:
   - stop = -1R;
   - full take = +2R.

4. adaptive_3R:
   - stop = -1R;
   - full take = +3R.

All policies:
- enter with the existing BASE modeled fill;
- use 1-second decision latency;
- trigger from completed one-second OHLC;
- fill the exit only at the first later eligible second open;
- if stop and take occur in the same completed second, mark ambiguous;
- if neither barrier triggers, force a time-cap exit after 30 minutes using the
  first later eligible second open;
- missing seconds never manufacture fills.

The current Nasdaq halt feed may remain unavailable; that limitation must be
reported explicitly and prevents any promotion claim from this request.

## Metrics

For every policy report:

- attempted episodes;
- closed / ambiguous / unresolved coverage;
- mean and day-balanced realized BASE net return;
- daily mean return;
- positive-return day count;
- positive-trade rate;
- mean winner / mean loser / payoff ratio;
- severe-loss rate <= -5%;
- mean realized R multiple:
    realized BASE return / stop distance;
- day-balanced realized R multiple;
- stop / take / time-cap exit rates;
- stop-distance distribution.

Matched comparisons:
- adaptive_2R minus fixed_tight;
- adaptive_3R minus fixed_runner.

Report matched mean and day-balanced return difference, R-multiple difference,
and number of days with positive matched improvement.

## Geometry bridge rule

An adaptive policy is worth carrying into the next entry-learning target only if
all hold:

1. closed coverage >= 90%;
2. matched day-balanced BASE return improves by at least +0.50 percentage
   points versus its fixed comparator;
3. matched day-balanced R-multiple improves by at least +0.20R;
4. adaptive policy beats the comparator's daily mean on at least 6 of 8
   calibration days;
5. severe-loss rate is not worse than the comparator.

This does not require raw HOT entries to become profitable. It asks only whether
the adaptive geometry is materially better and therefore deserves to define the
next learning target.

## Failure rule

If neither adaptive geometry passes, do not tune the 5-minute window, 3-8% cap,
2R/3R reward multiples, or support rule on calibration results.

Conclude that fixed -3% geometry is not the primary explanation for the missing
entry edge, and redesign the target around path value / abstention rather than
risk thresholds.
