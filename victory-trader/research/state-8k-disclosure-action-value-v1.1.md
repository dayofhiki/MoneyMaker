# State 8-K Disclosure Action Value v1.1 — pre-registration

## Status

Pre-registered after the Historical 8-K Disclosure Supply Probe v0.1 passed all
data-supply criteria and before implementing or viewing any v1.1 trading result.

Use only the already-seen January-March 2026 development state panels. Do not
load or inspect a fresh validation month.

## Hypothesis

Recent corporate disclosures may distinguish superficially similar runner states
using information that is independent from intraday price/volume and FINRA
short-volume behavior.

This branch tests 8-K metadata independently. It does not combine 8-K features
with the retired v0.9/v1.0 short-volume branches.

## Point-in-time rule

For a state on trading day D, use only 8-K rows with:

`D - 30 calendar days <= filing_date < D`.

Same-day filings are forbidden because date-only metadata does not establish
that a filing was public before the intraday decision time.

Never backfill a future filing.

## Fixed feature set

For every ticker-day construct exactly these features from strictly prior 8-K
disclosures:

1. `eight_k_unique_filings_30d`;
2. `eight_k_disclosures_30d`;
3. `eight_k_has_filing_7d`;
4. `eight_k_latest_filing_age_days`;
5. `eight_k_distinct_primary_categories_30d`;
6. `eight_k_primary_capital_and_financing_count_30d`;
7. `eight_k_primary_leadership_and_governance_count_30d`;
8. `eight_k_primary_shareholder_activity_count_30d`;
9. `eight_k_primary_strategic_transactions_count_30d`;
10. `eight_k_primary_financial_results_count_30d`;
11. `eight_k_primary_operations_and_strategy_count_30d`;
12. `eight_k_primary_risk_events_count_30d`;
13. `eight_k_primary_regulatory_and_compliance_count_30d`.

Counts and the 7-day flag are zero when no strictly prior filing exists.
Latest filing age is NaN when no prior filing exists.

The eight primary-category names are the complete primary-category vocabulary
observed by the outcome-blind supply probe. Their inclusion is based on metadata
availability, not realized trading outcomes.

Do not add secondary/tertiary categories, text embeddings, same-day filings, or
hand-selected categories after viewing v1.1 results.

## Models

Compare two leave-one-month-out policies.

### earliest_ev_cap1

Unchanged direct arithmetic base-net EV baseline with the existing causal
price/volume/breadth/sequence information set.

### eight_k_ev_cap1

Identical target, sampled training rows, calibration split, execution
assumptions, model capacity, and entry policy, with only the thirteen fixed 8-K
features appended.

Both use:

- 5/10/15-minute direct base-net EV targets;
- train-only 0.5%/99.5% target winsorization;
- chronological 80/20 fit/calibration split;
- HistGradientBoostingRegressor with squared error, learning_rate=0.05,
  max_iter=180, max_leaf_nodes=31, min_samples_leaf=300,
  l2_regularization=2.0;
- affine calibration;
- fixed predicted base-net EV gate +0.50%;
- first qualifying state per ticker-day;
- one attempted entry per ticker-day.

The first qualifying signal consumes the attempt even if the entry is
unfillable or the selected realized label is missing.

## Evaluation

Report for each held-out month:

- ticker-day enrichment coverage and filing prevalence;
- trades and trading days;
- gross, base-net, and stress-net means;
- day-balanced base-net mean;
- median, p05, positive rate, severe-loss rate, worst day;
- horizon mix and median entry minute;
- common-episode comparison versus baseline;
- pooled trading-day-cluster bootstrap for `eight_k_ev_cap1`.

## Promotion rule

Do not promote or consume a fresh month unless `eight_k_ev_cap1` satisfies all:

1. at least 15 trades in every month;
2. positive arithmetic base-net mean in every month;
3. positive day-balanced base-net mean in every month;
4. day-balanced base-net mean strictly above `earliest_ev_cap1` in every month;
5. base p05 and stress-net mean are not worse than baseline in any month;
6. 100% of scoreable ticker-days are successfully enriched with a completed
   prior-8-K query, including valid zero-filing rows; and
7. pooled trading-day-cluster bootstrap 95% lower bound is above zero.

If the exact branch fails, do not tune the +0.50% gate, 30-day window, category
subset, or category-specific thresholds using January-March outcomes.


## Result

Workflow request 37 completed successfully. Workflow request 38 repeated the
same pre-registered experiment with an equivalent market-wide 8-K retrieval
implementation. The two runs produced byte-equivalent metric tables for
month-by-month results, summaries, comparisons, scoreable coverage, and
enrichment coverage.

The bulk implementation reduced Massive REST requests for enrichment from 2,177
to 24 with zero retries while preserving identical feature coverage and trading
results.

### Enrichment

Scoreable ticker-day query completion was 100% in all three months.

| Month | Scoreable ticker-days | prior-30d 8-K prevalence | prior-7d prevalence |
|---|---:|---:|---:|
| 2026-01 | 2,751 | 42.06% | 16.25% |
| 2026-02 | 2,413 | 42.11% | 16.70% |
| 2026-03 | 2,761 | 40.71% | 16.70% |

### Month-by-month policy result

| Month | Policy | Trades | Base mean | Day-balanced | p05 | Stress mean | Severe-loss rate |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | baseline | 18 | -2.737% | -2.521% | -17.381% | -4.911% | 38.9% |
| 2026-01 | 8-K | 9 | -7.954% | -9.204% | -35.634% | -10.012% | 44.4% |
| 2026-02 | baseline | 35 | +2.833% | +4.673% | -9.215% | +0.727% | 20.0% |
| 2026-02 | 8-K | 32 | -0.629% | +0.007% | -14.696% | -2.663% | 25.0% |
| 2026-03 | baseline | 21 | +0.910% | -0.203% | -5.543% | -1.114% | 9.5% |
| 2026-03 | 8-K | 25 | +0.018% | -0.235% | -14.944% | -2.043% | 20.0% |

The 8-K branch also underperformed baseline on common ticker-day episodes in
every month:

- January: -4.554 percentage points across 8 common episodes;
- February: -2.766 points across 25;
- March: -2.724 points across 13.

Pooled day-balanced return was -1.622%, with a trading-day-cluster bootstrap
95% interval of [-4.511%, +0.695%].

### Pre-registered checks

Only 100% query-completion coverage passed. Trade-count sufficiency, positive
monthly arithmetic return, positive monthly day-balanced return, improvement
over baseline, tail/stress protection, and a positive bootstrap lower bound all
failed.

## Decision

Retire the exact v1.1 8-K metadata augmentation branch.

Do not tune the +0.50% gate, 30-day lookback, primary-category subset, or
category thresholds on January-March outcomes. The result indicates that coarse
recent-8-K metadata, despite good point-in-time coverage, does not stabilize the
current action-value policy.

Retain the optimized market-wide 8-K retrieval implementation because it is
point-in-time equivalent to the ticker-by-ticker implementation and reduces
network requests by roughly two orders of magnitude.

Do not consume a fresh validation month yet.
