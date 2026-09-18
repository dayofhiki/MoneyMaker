# Relative Short-Volume Context Probe v1.0 — pre-registration

## Status

Pre-registered before implementing or viewing any relative-context model result.

This probe is outcome-blind. It may inspect only the already-built January-March
2026 development state panels and their strictly lagged FINRA short-volume
features. It must not inspect realized-return labels to decide the formulation.

## Question

Is there enough causal contemporaneous runner cross-section to express each
ticker's lagged short-volume state relative to other runners that are already
observable at the same minute?

## Causal comparison set

At each exact `(trading_day, t)` state timestamp, use only rows already present
in the state panel at that timestamp. The state panel begins after a ticker has
causally crossed the runner threshold, so no later-in-the-day runner may be
backfilled into an earlier timestamp.

Do not group by the final set of tickers that eventually run that day.

## Candidate context features

Using only the eight already pre-registered v0.9 lagged short-volume features,
construct the following eight relative/context quantities:

1. `runner_other_mean_short_ratio_latest_prior`;
2. `runner_short_ratio_latest_minus_other_mean`;
3. `runner_short_ratio_latest_percentile`;
4. `runner_other_mean_short_ratio_mean5_prior`;
5. `runner_short_ratio_mean5_minus_other_mean`;
6. `runner_short_ratio_mean5_percentile`;
7. `runner_short_volume_latest_vs_mean5_minus_other_mean`;
8. `runner_finra_total_volume_latest_vs_mean5_minus_other_mean`.

"Other mean" is leave-one-out at the exact timestamp. Percentiles are computed
within the exact contemporaneous runner group, including the subject ticker, but
are set to NaN when fewer than two valid runners exist.

## Probe metrics

Report by month and overall:

- state rows;
- distribution of contemporaneous runner count;
- fraction of rows with at least 2, 3, and 5 runners;
- non-null coverage for each of the eight proposed context features.

Do not report realized returns, trading-policy results, or tune any trading gate.

## Go / no-go rule

A model branch is allowed only if, in every month:

1. at least 60% of scoreable state rows have a non-null latest-ratio percentile;
2. at least 60% have a non-null latest-ratio leave-one-out difference; and
3. the median contemporaneous runner count is at least 2.

If this fails, do not loosen the coverage rule using January-March returns.
