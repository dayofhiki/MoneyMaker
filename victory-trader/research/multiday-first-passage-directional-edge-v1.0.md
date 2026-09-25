# Request 202 — multi-day state first-passage directional edge

## Purpose

Request 201 showed that strictly-prior share supply materially improves ranking
for the rare +10%-before--3% runner event, but its selected P90/P95 tail remained
economically negative.

Do not tune Request 201 thresholds or combine supply with other branches yet.

Request 202 tests an independent causal hypothesis:

**a stock's recent multi-day behavioral history changes the meaning of the same
intraday surge state.**

The current entry models are rich in same-day regular-session information but
contain little direct context about whether the ticker has recently been a
runner, whether participation is expanding relative to its own history, whether
volatility is already exhausted, or where the current state sits relative to
strictly-prior multi-day highs/lows.

## Data split

No new dates.

- train: Request-171 FIT only;
- evaluate: Request-171 chronological CALIBRATION only;
- Request-178 remains sealed.

The first-passage execution/label contract is exactly Request 199:

- BASE friction;
- 1-second latency;
- 5-second entry expiry;
- stop barrier -3%;
- tight upside barrier +5%;
- runner upside barrier +10%;
- 30-minute observation cap.

## Strictly-prior daily source

For each unique ticker-day D, fetch unadjusted daily aggregates ending at D-1.
Use at most the previous 20 trading sessions.

No D daily bar may enter a feature. No current/latest endpoint is allowed.

At least five prior sessions are required for a populated multi-day feature
vector. Missing history remains missing and does not remove a state.

## Frozen multi-day feature family

Append exactly these 16 features to the Request-199 no-rich-second comparator:

1. hist_log_avg_volume_5d;
2. hist_log_avg_volume_20d;
3. hist_volume_ratio_5d_20d;
4. hist_log_avg_dollar_volume_5d;
5. hist_log_avg_dollar_volume_20d;
6. hist_realized_vol_5d_pct;
7. hist_realized_vol_20d_pct;
8. hist_avg_intraday_range_5d_pct;
9. hist_avg_intraday_range_20d_pct;
10. hist_runner_days_20d;
11. hist_big_down_days_20d;
12. hist_prior_close_vs_20d_high_pct;
13. hist_prior_close_vs_20d_low_pct;
14. state_vs_20d_high_pct;
15. state_vs_20d_low_pct;
16. hist_days_since_last_runner;

Definitions:

- daily return = close/previous daily close - 1;
- runner day = daily return >= +10%;
- big-down day = daily return <= -10%;
- realized volatility = population standard deviation of daily returns;
- intraday range = (high-low)/close;
- dollar volume proxy = close * volume;
- current state price is causal completed-state close;
- 20d high/low use only prior daily highs/lows;
- days-since-runner is trading-session distance within the available prior
  history, missing if no runner occurred.

No runner threshold other than the frozen +10%, no volume cutoff, no high/low
window, no feature subset, and no interaction may be chosen from calibration
outcomes.

## Models

For each barrier pair train two equal-day HGB classifiers with the same model
family and hyperparameters as Requests 199-201:

1. comparator: exact Request-199 no-rich-second feature family;
2. multi-day: comparator plus the frozen 16 multi-day features.

No score threshold or probability calibration is fitted.

## Required diagnostics

Report:

- history-query success;
- >=5-session support;
- >=20-session support;
- strict-prior max daily timestamp check;
- comparator and multi-day AUC / AP;
- AUC uplift;
- decisive-only AUC / AP;
- per-day AUC and day wins;
- P80/P90/P95 multi-day-score barrier diagnostics;
- all-state and P90 day-balanced barrier proxy.

## Development signal rule

A multi-day branch is worth converting into an executable action model only if
at least one barrier pair satisfies all:

1. multi-day AUC >= 0.62;
2. AUC uplift over comparator >= +0.03;
3. multi-day AUC exceeds comparator on at least 6 of 8 calibration days;
4. P90 day-balanced barrier proxy improves by at least +0.50 percentage points
   versus all states;
5. at least 90% of calibration states have >=5 strictly-prior daily sessions.

This is not a promotion rule. Request-178 remains sealed.

## Failure rule

If Request 202 fails, do not tune lookback windows, runner/down thresholds,
feature subsets, model capacity, barrier sizes, or score quantiles on
calibration outcomes.

Proceed to another genuinely different causal information source. Only if an
independent branch passes its own frozen signal rule should combination with the
Request-201 supply family be preregistered.


## Pre-result split-integrity amendment

Before any Request-202 outcome was read, the implementation audit identified a
semantic issue specific to low-priced small caps: unadjusted daily prices from
before and after a reverse/forward split cannot be compared as one continuous
price scale, while provider-adjusted historical prices can encode future
corporate actions.

The frozen solution is strictly causal:
- query split records only through D-1;
- if a split exists inside the daily lookback, discard all daily bars before the
  latest strictly-prior split execution date;
- compute the frozen multi-day features only from post-split bars;
- if split-history retrieval fails, expose no multi-day vector for that
  ticker-day rather than assuming no split;
- the 90% >=5-session support rule remains unchanged and split-query success
  must be complete.

No outcome, model metric, threshold, date, or feature definition was inspected
or changed by this amendment.
