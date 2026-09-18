# Historical 8-K Disclosure Supply Probe v0.1 — pre-registration

## Status

Pre-registered before implementation and before viewing any probe result.

This is a point-in-time data-supply probe only. It uses the already-seen
January-March 2026 development candidate panels and must not inspect a fresh
validation month.

## Question

Can SEC Form 8-K disclosure metadata provide a sufficiently available,
causally-safe information source for small-cap runner research that is
independent from price/volume and FINRA short-volume features?

## Source

Massive Stocks REST endpoint:

`GET /stocks/filings/8-K/vX/disclosures`

The endpoint returns SEC filing date, accession number, ticker mapping, and
classified disclosure categories.

## Sampling

Use the same deterministic candidate-v2 signal construction as the short-volume
supply probe. Within each held-out month, sort candidate signals by trading day,
threshold, ticker, and timestamp, then select 25 approximately evenly spaced
events when at least 25 exist. Maximum sample: 75 events.

Selection must not use later realized returns.

## Point-in-time rule

For an event on trading day D:

- query only filings from D-30 calendar days through D-1 calendar day;
- reject the probe if any returned `filing_date >= D`;
- never use a same-day filing because date-only metadata cannot prove the filing
  was public before the intraday decision;
- never backfill from a later filing.

## Metrics

For each sampled event report:

- unique 8-K filings in the prior 30 days;
- disclosures in the prior 30 days;
- whether at least one filing exists within 7 and 30 days;
- age of the latest prior filing;
- number of distinct primary, secondary, and tertiary categories;
- whether retrieved disclosures contain non-empty category labels.

Aggregate coverage by month and overall, plus the most frequent primary,
secondary, and tertiary categories.

## Go / no-go rule

The source is sufficiently available to justify a later pre-registered model
branch only if:

1. at least 20% of sampled events have at least one prior 8-K filing within 30
   calendar days;
2. every development month individually has at least 10% prior-30-day filing
   coverage; and
3. at least 80% of retrieved disclosure rows have non-empty primary, secondary,
   and tertiary category labels.

These are data-supply criteria only, not trading-performance criteria.

If the probe passes, any later model branch must pre-register its filing feature
set before looking at trading results. If it fails, do not widen the 30-day
window or use same-day filings based on January-March outcomes.


## Result

Workflow request 34 completed successfully.

- sampled events: 75 across January-March 2026;
- unique tickers: 71;
- retrieved disclosure rows: 88;
- primary / secondary / tertiary category completeness: 100%.

Coverage:

| Month | 7-day filing coverage | 30-day filing coverage |
|---|---:|---:|
| 2026-01 | 20% | 40% |
| 2026-02 | 28% | 48% |
| 2026-03 | 20% | 44% |
| Overall | 22.67% | 44% |

All three pre-registered data-supply rules passed. The most frequent primary
categories included capital_and_financing, leadership_and_governance,
shareholder_activity, and strategic_transactions.

## Decision

Retain strictly prior 8-K disclosure metadata as a viable future information
source. It is materially sparser than FINRA short-volume data, so do not mix it
into the current relative-short-volume experiment. Any trading-model use must be
pre-registered separately and must preserve the rule `filing_date < trading_day`.
