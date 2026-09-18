# State Multi-Source Interaction Action Value v1.3 — pre-registration

## Status

Pre-registered after the publication-safe Short-Interest Position Probe v1.2
passed all supply rules and before implementing or viewing any v1.3 trading
result.

Use only the already-seen January-March 2026 development state panels. Do not
load or inspect a fresh validation month.

This is an adaptive development hypothesis. January-March have already informed
the choice to test interactions, so even a pass here is not final evidence of
edge. A pass only authorizes one pre-registered fresh-month validation.

## Hypothesis

The previous single-family augmentation branches may fail because the useful
market structure is conditional across information sources. A high short
position, recent short-sale flow, corporate-event state, and intraday demand may
only become informative jointly.

Test whether one fixed interaction-capable model using all causal information
families can convert the previously observed relative timing structure into a
robust absolute trading policy.

## Point-in-time rules

All existing core state features remain causal.

### FINRA short volume

Use the exact v0.9 eight-feature definition. Every source record must satisfy:

`record_date < state_trading_day`.

### FINRA short interest

Use only reports whose official FINRA publication date is strictly before the
state trading day. Settlement date is never treated as an availability
timestamp. Same-publication-day reports are forbidden.

Use the fixed 2025-11 through 2026-03 FINRA settlement/publication schedule from
the v1.2 probe.

### SEC 8-K

Use only disclosures with:

`state_day - 30 days <= filing_date < state_day`.

Same-day filings remain forbidden.

No final-day runner universe, future return, future HOD, future filing, future
short-volume record, or unpublished short-interest report may enter a feature.

## Frozen external feature families

### A. Short volume — exact 8 v0.9 features

1. `short_ratio_latest_prior`
2. `short_ratio_mean5_prior`
3. `short_ratio_mean20_prior`
4. `short_ratio_latest_minus_mean5`
5. `short_ratio_mean5_minus_mean20`
6. `short_volume_latest_vs_mean5`
7. `finra_total_volume_latest_vs_mean5`
8. `short_volume_latest_age_days`

### B. 8-K — exact 13 v1.1 features

1. `eight_k_unique_filings_30d`
2. `eight_k_disclosures_30d`
3. `eight_k_has_filing_7d`
4. `eight_k_latest_filing_age_days`
5. `eight_k_distinct_primary_categories_30d`
6. `eight_k_primary_capital_and_financing_count_30d`
7. `eight_k_primary_leadership_and_governance_count_30d`
8. `eight_k_primary_shareholder_activity_count_30d`
9. `eight_k_primary_strategic_transactions_count_30d`
10. `eight_k_primary_financial_results_count_30d`
11. `eight_k_primary_operations_and_strategy_count_30d`
12. `eight_k_primary_risk_events_count_30d`
13. `eight_k_primary_regulatory_and_compliance_count_30d`

### C. Short interest — fixed 8 publication-safe features

1. `short_interest_latest`
2. `short_interest_avg_daily_volume_latest`
3. `short_interest_days_to_cover_latest`
4. `short_interest_change_pct`
5. `short_interest_avg_daily_volume_change_pct`
6. `short_interest_days_to_cover_change`
7. `short_interest_publication_age_days`
8. `short_interest_reports_available`

Do not add, remove, hand-select, threshold, rank, or transform external features
after viewing v1.3 outcomes.

## Models

Compare exactly two leave-one-month-out policies.

### earliest_ev_cap1

The unchanged direct arithmetic base-net EV baseline using the existing causal
core price/volume/breadth/sequence state.

### multi_source_ev_cap1

Identical to the baseline in target, training rows, model capacity, calibration,
execution assumptions, and entry policy, except that all 29 frozen external
features above are appended simultaneously.

The HistGradientBoostingRegressor is intentionally retained because tree splits
can represent conditional interactions without manually specifying
outcome-selected crosses.

Both policies use:

- 5/10/15-minute direct base-net EV targets;
- train-only 0.5%/99.5% target winsorization;
- chronological 80/20 fit/calibration split;
- squared-error HistGradientBoostingRegressor;
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=31;
- min_samples_leaf=300;
- l2_regularization=2.0;
- the same affine calibration;
- fixed predicted base-net EV gate of +0.50%;
- first qualifying state per ticker-day;
- one attempted entry per ticker-day.

The first qualifying signal consumes the attempt even when the selected entry is
unfillable or its selected realized label is missing.

## Evaluation

For every held-out development month report:

- trades and trading days;
- gross/base/stress mean;
- day-balanced base mean;
- median, p05, positive rate, severe-loss rate, worst day;
- horizon mix and median entry minute;
- common-episode comparison versus baseline;
- scoreable-state coverage for each external family;
- pooled trading-day-cluster bootstrap for `multi_source_ev_cap1`.

The recorded v0.9 short-volume branch is a diagnostic historical comparator, not
a promotion threshold.

## Promotion rule

The branch passes development only if ALL conditions hold:

1. at least 15 trades in every month;
2. positive arithmetic base-net mean in every month;
3. positive day-balanced base-net mean in every month;
4. day-balanced base-net mean strictly above `earliest_ev_cap1` in every
   month;
5. base p05 and stress-net mean are not worse than baseline in any month;
6. every month has >=90% scoreable-state coverage for
   `short_ratio_latest_prior`;
7. every month has >=90% scoreable-state coverage for
   `short_interest_latest`, and 100% completed 8-K query coverage;
8. pooled trading-day-cluster bootstrap 95% lower bound for day-balanced
   base-net return is above zero.

If all eight pass, freeze the branch and pre-register exactly one unseen
validation month before accessing it.

If the branch fails, do not tune the +0.50% gate, model capacity, external
feature subset, short-interest thresholds, 8-K categories, or short-volume
windows on January-March. Treat the combined-feature hypothesis as failed and
move the next research branch to decision formulation rather than additional
feature-family search.


## Result

Workflow request 41 completed successfully. The exact pre-registered
multi-source interaction branch failed development promotion.

### Coverage

All external families were dense enough for the test:

- short-volume latest-prior coverage: 100% in January, February, and March;
- publication-safe short-interest latest coverage: 98.71%, 98.94%, and 98.90%;
- 8-K query completion: 100% every month;
- short-interest query completion: 100% every month.

### Month-by-month

| Month | Policy | Trades | Base mean | Day-balanced | p05 | Stress mean | Severe-loss rate | Worst day |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-01 | baseline | 18 | -2.737% | -2.521% | -17.381% | -4.911% | 38.9% | -36.154% |
| 2026-01 | multi-source | 11 | +0.037% | +1.442% | -10.825% | -2.098% | 36.4% | -9.128% |
| 2026-02 | baseline | 35 | +2.833% | +4.673% | -9.215% | +0.727% | 20.0% | -9.205% |
| 2026-02 | multi-source | 15 | -0.479% | -0.666% | -9.792% | -2.556% | 26.7% | -10.582% |
| 2026-03 | baseline | 21 | +0.910% | -0.203% | -5.543% | -1.114% | 9.5% | -27.603% |
| 2026-03 | multi-source | 16 | -0.440% | +0.053% | -11.525% | -2.480% | 12.5% | -4.987% |

Pooled multi-source day-balanced return was +0.120%, with a trading-day-cluster
bootstrap 95% interval of [-2.131%, +2.417%].

Only the two data-coverage criteria passed. Trade-count sufficiency, positive
monthly arithmetic return, positive monthly day-balanced return, improvement
over baseline every month, tail/stress protection, and positive bootstrap lower
bound all failed.

### Selection diagnostic

The model did not merely make a stable conservative subset.

- January removed 11 baseline-only ticker-days whose baseline mean was -1.676%,
  but its 4 newly selected ticker-days also averaged -1.896%.
- February removed 23 baseline-only ticker-days averaging +2.986% and added
  3 ticker-days averaging -2.255%.
- March removed 16 baseline-only ticker-days averaging -0.210% but added
  11 ticker-days averaging -3.129%.

This is consistent with unstable absolute action-value estimation rather than a
missing-data problem.

## Decision

Retire the exact v1.3 squared-error multi-source interaction branch.

Do not search external-feature subsets, interactions, thresholds, model
capacities, or entry gates on January-March.

The fixed multi-source information set remains frozen for the next branch so
that the next experiment changes decision formulation rather than searching for
more data. The next development question is whether a robust typical-outcome
target, rather than squared-error expected-return regression, can use the same
information more stably across regimes.

Do not consume a fresh validation month.
