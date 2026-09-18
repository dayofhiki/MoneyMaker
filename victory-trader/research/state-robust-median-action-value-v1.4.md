# State Robust Median Action Value v1.4 — pre-registration

## Status

Pre-registered after the exact v1.3 multi-source squared-error branch failed and
before implementing or viewing any v1.4 trading result.

Use only the already-seen January-March 2026 development state panels. Do not
load or inspect a fresh validation month.

The external information set is frozen exactly as in v1.3. This branch changes
the decision target, not the data search.

## Hypothesis

Small-cap runner returns are heavy-tailed. Squared-error regression can be
dominated by rare large outcomes and may produce unstable absolute expected
return estimates across regimes even when useful ordering information exists.

Test whether predicting the conditional median after-cost return for each
5/10/15-minute BUY action produces a more stable entry policy using the same
causal core and the same 29 frozen external features.

## Frozen information set

Use exactly:

- the existing causal core price/volume/breadth/sequence state;
- the exact 8 v0.9 short-volume features;
- the exact 13 v1.1 8-K features;
- the exact 8 publication-safe v1.3 short-interest features.

No external feature may be added, removed, ranked, thresholded, crossed by hand,
or transformed based on January-March outcomes.

All point-in-time rules from v1.3 remain unchanged.

## Baseline

`earliest_ev_cap1` remains the unchanged squared-error direct arithmetic
base-net EV baseline using core features and the fixed +0.50% expected-EV gate.

## Primary policy: median_multi_source_cap1

For each horizon 5, 10, and 15 minutes:

- target is the same realized base-net return used by the existing direct EV
  model;
- training rows and chronological 80/20 fit/calibration split are unchanged;
- apply train-only 0.5%/99.5% target winsorization exactly as before;
- use HistGradientBoostingRegressor with `loss="absolute_error"`;
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=31;
- min_samples_leaf=300;
- l2_regularization=2.0;
- random seed remains `20260921 + horizon`.

The model output is interpreted as conditional median base-net return.

### Robust calibration

Do not fit the existing least-squares affine calibration, because that would
reintroduce mean/outlier sensitivity.

On the chronological calibration split, use only an additive median-bias
correction:

`calibrated_median = raw_prediction + median(actual - raw_prediction)`.

If fewer than 100 finite calibration observations exist, use zero correction.

No multiplicative slope is fitted.

### Action rule

At every scoreable state:

1. predict calibrated median base-net return for BUY_5M, BUY_10M, BUY_15M;
2. select the horizon with the highest predicted median;
3. signal only when that highest predicted median is >= +0.50%;
4. use the first qualifying state per ticker-day;
5. one attempted entry per ticker-day;
6. the first signal consumes the attempt even if unfillable or the selected
   realized label is missing.

The +0.50% gate is deliberately unchanged. Do not tune it for a median target on
January-March.

## Evaluation

Report for baseline and primary policy:

- trades, ticker-day episodes, trading days;
- gross/base/stress means;
- day-balanced base mean;
- median, p05, positive rate, severe-loss rate, worst day;
- horizon mix and median entry minute;
- common-episode comparison;
- external family coverage;
- pooled trading-day-cluster bootstrap for `median_multi_source_cap1`.

## Promotion rule

The primary policy passes development only if ALL conditions hold:

1. at least 15 trades in every month;
2. positive arithmetic base-net mean in every month;
3. positive day-balanced base-net mean in every month;
4. day-balanced base-net mean strictly above `earliest_ev_cap1` in every
   month;
5. base p05 and stress-net mean are not worse than baseline in any month;
6. short-volume latest-prior coverage >=90% in every month;
7. publication-safe short-interest latest coverage >=90% and 8-K query
   completion =100% in every month;
8. pooled trading-day-cluster bootstrap 95% lower bound for day-balanced
   base-net return is above zero.

If all eight pass, freeze this exact branch and pre-register one unseen
validation month before accessing it.

If the branch fails, do not tune the median gate, quantile, loss, calibration
rule, model capacity, or external feature subset on January-March. The next
branch must change the action/WAIT formulation rather than continue target or
feature tweaking.


## Result

Workflow request 42 completed successfully. The exact pre-registered robust
median branch failed development promotion.

### Coverage

Data coverage was sufficient and essentially identical to v1.3:

- short-volume latest-prior coverage: 100% in January, February, and March;
- publication-safe short-interest latest coverage: 98.71%, 98.94%, and 98.90%;
- 8-K query completion: 100% every month;
- short-interest query completion: 100% every month.

### Trading result

The primary policy `median_multi_source_cap1` generated zero trades in all
three development months.

No scoreable state in January, February, or March produced a highest calibrated
conditional-median base-net prediction of at least +0.50% across the 5/10/15
minute actions.

The unchanged baseline remained:

| Month | Baseline trades | Base mean | Day-balanced |
|---|---:|---:|---:|
| 2026-01 | 18 | -2.737% | -2.521% |
| 2026-02 | 35 | +2.833% | +4.673% |
| 2026-03 | 21 | +0.910% | -0.203% |

Because the primary policy produced no trades, its pooled day-balanced estimate
and bootstrap interval are undefined.

Only the two external-data coverage criteria passed. All trading-performance
criteria failed.

## Interpretation

This failure is materially different from v1.3.

The squared-error mean-return model was willing to assign >+0.50% expected
after-cost value to some states, but the robust conditional-median model was not
willing to assign >+0.50% typical after-cost return to any state.

That is consistent with a heavy-tailed payoff structure in which occasional
large winners can lift conditional means while the typical outcome remains too
small or negative after costs. It also explains why relative within-episode
timing can be detectable without yielding a stable absolute +EV entry threshold.

This does not prove that every conditional median is negative; it proves only
that, under the exact pre-registered model and robust calibration, none reached
the fixed +0.50% action hurdle.

## Decision

Retire the exact v1.4 robust-median +0.50% branch.

Do not lower or tune the median gate, change the quantile, alter the robust
calibration rule, or search feature subsets on January-March.

The fixed multi-source information set remains available, but the next research
branch must change the action/WAIT formulation rather than continue feature or
target-threshold tuning.

No fresh validation month was consumed.
