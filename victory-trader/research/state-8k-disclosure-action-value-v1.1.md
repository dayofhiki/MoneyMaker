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
