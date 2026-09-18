# State Short-Volume Action Value v0.9 — pre-registration

## Status

Pre-registered after Historical Short-Volume Supply Probe v0.1 passed all
coverage criteria and before implementation or viewing any v0.9 model result.

Use only the already-seen January-March 2026 development state panels. Do not
load or inspect a fresh validation month.

## Motivation

The existing causal price/volume state model remains regime-unstable: January
loses while February and March are more favorable. A separate severe-loss
classifier, relative-timing stopping rules, rank-residual fusion, and a
log-growth objective all failed their pre-registered promotion rules.

The short-volume supply probe found dense historical FINRA short-sale activity:
100% of 75 sampled development events had a prior record within seven calendar
days and 100% had at least five prior records. This branch tests whether that
new, causally lagged information improves the existing action-value model.

## Point-in-time data construction

For each state trading day D, short-volume features may use only records with
`record_date < D`.

Build a reusable daily whole-market cache using
`GET /stocks/v1/short-volume?date=YYYY-MM-DD`. The enrichment process must
reject any merge that would use a record dated D or later.

No same-day short volume is allowed because the daily record is not complete at
intraday decision time.

## Fixed added features

For each ticker-day, compute exactly these eight features from prior FINRA
records:

1. `short_ratio_latest_prior`: latest prior short-volume ratio;
2. `short_ratio_mean5_prior`: mean ratio over the latest five prior records;
3. `short_ratio_mean20_prior`: mean ratio over the latest twenty prior records;
4. `short_ratio_latest_minus_mean5`;
5. `short_ratio_mean5_minus_mean20`;
6. `short_volume_latest_vs_mean5`: latest short volume divided by the latest
   five-record mean short volume;
7. `finra_total_volume_latest_vs_mean5`: latest total FINRA-reported volume
   divided by the latest five-record mean total volume; and
8. `short_volume_latest_age_days`: calendar-day age of the latest prior record.

Ratios 6 and 7 are NaN if the denominator is non-positive. Means are NaN unless
the full requested history length is available: five records for five-record
features and twenty records for twenty-record features.

Do not add market cap, float, news, short interest, current-day short volume, or
any feature selected after viewing v0.9 outcomes.

## Model

Compare two leave-one-month-out policies.

### earliest_ev_cap1

Unchanged direct arithmetic base-net EV baseline:

- existing causal feature family;
- existing 5/10/15-minute direct-EV regressors;
- train-only 0.5%/99.5% target winsorization;
- same chronological 80/20 fit/calibration split;
- same HGB model capacity and affine calibration;
- fixed +0.50% predicted base-net EV entry gate;
- first qualifying state per ticker-day; one attempted entry.

### short_volume_ev_cap1

Identical training target, training rows, calibration, model hyperparameters,
execution assumptions, +0.50% entry gate, and one-attempt rule, but append the
eight fixed lagged short-volume features above to the model input.

The first qualifying signal consumes the ticker-day attempt even if unfillable
or missing the selected realized label.

## Evaluation

Leave one month out, train on the other two, and repeat for January, February,
and March.

Report:

- data enrichment coverage by month;
- trades, days, ticker-day episodes;
- gross, base-net, stress-net means;
- day-balanced base-net mean;
- median, p05, severe-loss rate, worst day;
- action-horizon mix and median entry minute;
- common-episode entry delay and realized-return delta;
- pooled trading-day-cluster bootstrap for `short_volume_ev_cap1`.

## Success rule

Do not promote or consume a fresh month unless `short_volume_ev_cap1`:

1. has at least 15 trades in every month;
2. has positive base-net mean in every month;
3. has positive day-balanced base-net mean in every month;
4. improves day-balanced base-net mean over `earliest_ev_cap1` in every month;
5. does not worsen base p05 or stress-net mean versus baseline in any month;
6. has at least 90% non-null coverage for `short_ratio_latest_prior` in every
   month's scoreable states; and
7. has a pooled trading-day-cluster bootstrap 95% lower bound above zero.

All seven conditions must pass.

If this exact branch fails, do not tune the +0.50% gate, short-ratio thresholds,
rolling-window lengths, or manually select a subset of the eight features using
January-March outcomes. Retire the exact branch or change the information set /
decision formulation.
