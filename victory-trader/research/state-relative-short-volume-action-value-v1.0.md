# State Relative Short-Volume Action Value v1.0 — pre-registration

## Status

Pre-registered after the outcome-blind Relative Short-Volume Context Probe v1.0
passed all coverage rules and before implementing or viewing any v1.0 trading
result.

Use only the already-seen January-March 2026 development state panels. Do not
load or inspect a fresh validation month.

## Hypothesis

The v0.9 absolute short-volume branch materially improved January but degraded
February and March. The next hypothesis is not that the v0.9 thresholds were
wrong. It is that an absolute FINRA short-volume value may have different
meaning depending on the contemporaneous runner environment.

The model therefore receives the same strictly lagged ticker-level short-volume
features plus fixed causal cross-sectional context at the exact state timestamp.

## Point-in-time rules

All v0.9 FINRA records must still satisfy:

`short_volume_record_date < state_trading_day`.

Relative context is computed only from rows that exist at the exact
`(trading_day, t)` timestamp in the state panel. A ticker that becomes a runner
later in the day must not be inserted into an earlier comparison set.

No final-day runner universe, same-day FINRA record, future return, future HOD,
or future filing information may enter any feature.

## Fixed short-volume feature set

### Eight absolute v0.9 features

1. `short_ratio_latest_prior`;
2. `short_ratio_mean5_prior`;
3. `short_ratio_mean20_prior`;
4. `short_ratio_latest_minus_mean5`;
5. `short_ratio_mean5_minus_mean20`;
6. `short_volume_latest_vs_mean5`;
7. `finra_total_volume_latest_vs_mean5`;
8. `short_volume_latest_age_days`.

### Eight relative/context features

1. `runner_other_mean_short_ratio_latest_prior`;
2. `runner_short_ratio_latest_minus_other_mean`;
3. `runner_short_ratio_latest_percentile`;
4. `runner_other_mean_short_ratio_mean5_prior`;
5. `runner_short_ratio_mean5_minus_other_mean`;
6. `runner_short_ratio_mean5_percentile`;
7. `runner_short_volume_latest_vs_mean5_minus_other_mean`;
8. `runner_finra_total_volume_latest_vs_mean5_minus_other_mean`.

Do not add, delete, transform, or hand-select features after viewing v1.0
outcomes.

## Models and policies

Use the unchanged leave-one-month-out framework.

### earliest_ev_cap1

Unchanged direct arithmetic base-net EV baseline with no FINRA features.

### short_volume_ev_cap1

The exact v0.9 absolute short-volume model, rerun as a control.

### relative_short_volume_ev_cap1

Identical to `short_volume_ev_cap1` except that the eight fixed relative/context
features are appended to the model input.

All three policies use:

- the same existing causal price/volume/breadth/sequence feature family;
- 5/10/15-minute direct base-net EV targets;
- train-only 0.5%/99.5% target winsorization;
- chronological 80/20 fit/calibration split;
- HistGradientBoostingRegressor with squared error, learning_rate=0.05,
  max_iter=180, max_leaf_nodes=31, min_samples_leaf=300,
  l2_regularization=2.0;
- the same affine calibration;
- fixed predicted base-net EV gate of +0.50%;
- first qualifying state per ticker-day;
- one attempted entry per ticker-day;
- identical execution-cost assumptions.

The first qualifying signal consumes the ticker-day attempt even if the selected
entry is unfillable or the selected realized label is missing.

## Evaluation

For every held-out month report:

- trades, trading days, ticker-day episodes;
- gross, base-net, and stress-net means;
- day-balanced base-net mean;
- median, p05, positive rate, severe-loss rate, worst day;
- horizon mix and median entry minute;
- common-episode comparisons versus baseline and versus v0.9 absolute model;
- scoreable-state context coverage.

Also report a pooled trading-day-cluster bootstrap for
`relative_short_volume_ev_cap1`.

## Promotion rule

Do not promote or consume a fresh month unless
`relative_short_volume_ev_cap1` satisfies all seven conditions:

1. at least 15 trades in every month;
2. positive arithmetic base-net mean in every month;
3. positive day-balanced base-net mean in every month;
4. day-balanced base-net mean strictly above `earliest_ev_cap1` in every month;
5. base p05 and stress-net mean are not worse than `earliest_ev_cap1` in any
   month;
6. latest-ratio relative-context coverage is at least 90% in every month's
   scoreable states; and
7. pooled trading-day-cluster bootstrap 95% lower bound for day-balanced
   base-net return is above zero.

Comparison against `short_volume_ev_cap1` is diagnostic and must be reported,
but is not an extra promotion criterion. This avoids defining success around the
known month-specific v0.9 failures.

If the exact branch fails, do not tune the +0.50% gate, cross-sectional
percentile thresholds, runner-count cutoffs, rolling windows, or feature subset
on January-March outcomes.
