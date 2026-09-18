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


## Result

Workflow request 33 completed successfully.

Enrichment:

- 2,177 tickers had FINRA short-volume history;
- 95 whole-market daily requests, zero retries;
- latest-prior short-volume ratio coverage was 100% in all three months;
- five-record feature coverage was approximately 99.3%, 99.8%, and 99.5%;
- twenty-record feature coverage was approximately 98.5%, 98.4%, and 98.3%.

Policy results:

| Month | Baseline base mean | Short-volume base mean | Baseline day-balanced | Short-volume day-balanced |
|---|---:|---:|---:|---:|
| 2026-01 | -2.737% | +1.245% | -2.521% | +2.861% |
| 2026-02 | +2.833% | +1.213% | +4.673% | +0.992% |
| 2026-03 | +0.910% | +0.384% | -0.203% | -0.551% |

The short-volume branch produced at least 22 trades in every month and achieved a
positive arithmetic base-net mean in all three months for the first time in this
research sequence. However, it did not produce a positive day-balanced mean in
March, did not improve the baseline day-balanced mean in every month, worsened
some p05/stress metrics, and its pooled day-cluster bootstrap interval remained
wide with a 95% lower bound of -1.508%.

Pre-registered checks passed only:

- enough trades every month;
- positive base-net mean every month;
- >=90% short-volume coverage every month.

The exact branch therefore failed promotion.

## Decision

Retire the exact eight-feature `short_volume_ev_cap1` branch and do not tune its
gate, rolling windows, or feature subset on January-March outcomes.

Retain the factual finding that strictly lagged FINRA short-volume information is
dense and materially changes out-of-month behavior, including flipping January
from negative to positive while preserving positive arithmetic means in all
three development months. Treat this as evidence that additional non-price
information is worth investigating, not as a validated trading edge.

Do not consume a fresh validation month yet.


## Result

Workflow request 33 completed successfully. The exact pre-registered branch
failed promotion.

### Data coverage

Lagged FINRA short-volume enrichment was dense:

- January: 602,356 rows / 3,862 ticker-days; latest-prior ratio coverage 100.0%;
- February: 529,699 rows / 3,603 ticker-days; latest-prior ratio coverage 100.0%;
- March: 604,665 rows / 4,089 ticker-days; latest-prior ratio coverage 100.0%;
- five-record feature coverage was 99.28%, 99.77%, and 99.49%;
- twenty-record feature coverage was 98.49%, 98.41%, and 98.28%.

The enrichment used 95 Massive REST requests with zero retries and enforced
`record_date < trading_day`.

### Month-by-month policy result

| Month | Policy | Trades | Base mean | Day-balanced | p05 | Stress mean | Severe loss rate | Worst day |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-01 | baseline | 18 | -2.737% | -2.521% | -17.381% | -4.911% | 38.9% | -36.154% |
| 2026-01 | short-volume | 24 | +1.245% | +2.861% | -11.842% | -0.983% | 20.8% | -7.416% |
| 2026-02 | baseline | 35 | +2.833% | +4.673% | -9.215% | +0.727% | 20.0% | -9.205% |
| 2026-02 | short-volume | 22 | +1.213% | +0.992% | -9.102% | -0.817% | 27.3% | -9.205% |
| 2026-03 | baseline | 21 | +0.910% | -0.203% | -5.543% | -1.114% | 9.5% | -27.603% |
| 2026-03 | short-volume | 25 | +0.384% | -0.551% | -7.617% | -1.794% | 12.0% | -18.724% |

The short-volume model is the first tested branch in this line to keep at least
15 trades and a positive arithmetic base mean in all three development months.
However, it did not achieve positive day-balanced return in March, did not
improve the day-balanced baseline in February or March, and worsened tail/stress
metrics in some months.

Pooled day-balanced mean was +0.838%, but the day-cluster bootstrap 95% interval
was [-1.508%, +3.374%].

### Selection diagnostic

The failure is strongly regime-dependent rather than a simple lack of signal.

- January: the short-volume policy removed 4 baseline ticker-days whose baseline
  mean was -4.990%; on 14 common ticker-days it improved realized return by
  +4.874 percentage points on average.
- February: it removed 15 baseline ticker-days whose baseline mean was +4.776%;
  common-ticker-day realized return fell by 0.471 points on average.
- March: it removed 15 baseline ticker-days whose baseline mean was near flat
  (+0.017%); 19 added ticker-days were also near flat (+0.011%), while common
  ticker-day realized return fell by 1.579 points on average.

January's common-episode improvement is partly concentrated: one SXTC episode
changed from -36.154% under the baseline entry to +14.383% under the augmented
entry. Removing that single common-episode delta still leaves a positive average
January common-episode improvement, but much smaller.

### Pre-registered checks

Passed:

1. at least 15 trades every month;
2. positive base-net mean every month;
3. short-volume feature coverage at least 90% every month.

Failed:

1. positive day-balanced mean every month;
2. day-balanced improvement over baseline every month;
3. no worsening of p05/stress every month;
4. pooled bootstrap lower bound above zero.

## Decision

Retire the exact v0.9 absolute-feature augmentation branch. Do not tune the
+0.50% entry gate, rolling windows, or feature subset on January-March.

The result does not support discarding short-volume information entirely. It
shows a material January risk-control effect but an opposite selection effect in
February, consistent with a regime-dependent interpretation of the same
short-volume values. Any next branch should therefore change the formulation,
not threshold-tune v0.9. A defensible next question is whether short-volume
features should be expressed relative to their contemporaneous market /
small-cap-runner cross-section, or conditioned on an independently defined
market regime, with the formulation pre-registered before outcomes are viewed.
