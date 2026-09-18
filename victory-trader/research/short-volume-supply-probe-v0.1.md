# Historical Short-Volume Supply Probe v0.1 — pre-registration

## Status

Pre-registered before implementation and before viewing any probe result.

This is a data-supply and point-in-time safety probe only. It uses the
already-seen January-March 2026 development candidate panels and must not inspect
a fresh validation month.

## Question

Can FINRA daily short-volume history provide a dense, causally available
information source for small-cap runner research that is genuinely different
from the current price/volume state features?

## Source

Massive Stocks REST endpoint:

`GET /stocks/v1/short-volume`

The endpoint reports daily FINRA short-sale volume from off-exchange venues and
ATS facilities. Only records strictly before the candidate trading day are
allowed.

## Sampling

Reuse the existing candidate-v2 signal construction independently inside each
held-out development month. From each month, sort signals deterministically by
trading day, threshold, ticker, and timestamp, then take 25 approximately evenly
spaced rows when at least 25 signals exist. Maximum probe size: 75 candidate
events.

No event may be selected using its later realized return.

## Point-in-time rule

For an event on trading day D:

- request only dates from D-30 calendar days through D-1 calendar day;
- sort returned records by date;
- reject the probe with an exception if any returned record has date >= D;
- never use the event day's short-volume record;
- never backfill with a future record.

This probe treats the dataset's reported `date` as the trade-activity date.
A production feature branch must still use lagged records only.

## Metrics

For each sampled event report:

- number of prior short-volume records in the 30-calendar-day request;
- age in calendar days of the latest prior record;
- whether a prior record exists within 1, 3, and 7 calendar days;
- latest short-volume ratio;
- mean short-volume ratio over the latest five available records;
- latest-minus-five-record-mean ratio;
- latest short volume and total reported volume.

Aggregate coverage separately by held-out month and overall.

## Go / no-go rule

This probe is considered sufficiently dense to justify a pre-registered model
branch only if:

1. at least 90% of sampled events have a prior short-volume record no more than
   7 calendar days old;
2. every development month individually has at least 80% such coverage; and
3. at least 80% of sampled events have five prior short-volume records in the
   30-calendar-day window.

These are coverage criteria, not trading-performance criteria.

If the probe passes, the next branch may pre-register lagged short-volume
features and test them in the existing leave-one-month-out framework. If it
fails, do not tune the age window or select only tickers with available data
using January-March outcomes.


## Result

Workflow request 32 completed successfully.

Sample:

- 75 candidate events across January-March 2026;
- 71 unique tickers;
- 75 REST requests, zero retries.

Coverage:

| Month | Sampled | prior <=1d | prior <=3d | prior <=7d | five-record coverage | median latest age | median 30d records |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-01 | 25 | 64.0% | 100.0% | 100.0% | 100.0% | 1 day | 19 |
| 2026-02 | 25 | 84.0% | 96.0% | 100.0% | 100.0% | 1 day | 21 |
| 2026-03 | 25 | 76.0% | 100.0% | 100.0% | 100.0% | 1 day | 21 |
| Overall | 75 | 74.7% | 98.7% | 100.0% | 100.0% | 1 day | 20 |

All three pre-registered coverage criteria passed.

## Decision

Promote lagged FINRA short-volume data from supply probe to a pre-registered
model-feature branch. Preserve the strict rule that only records with
`date < trading_day` may be used. Do not use same-day short volume.
