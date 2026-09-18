# Historical Short-Interest Position Probe v1.2 — pre-registration

## Status

Pre-registered before implementation and before viewing any short-interest probe
result.

This is a data-supply and point-in-time safety probe only. It uses the
already-seen January-March 2026 development candidate panels and must not inspect
a fresh validation month.

## Question

Can FINRA short-interest position data provide a sufficiently dense, causally
available information source for small-cap runner research that is distinct from
daily short-volume activity?

## Source

Massive Stocks REST endpoint:

`GET /stocks/v1/short-interest`

Short interest is an outstanding-position measure reported twice per month. It
must not be treated as equivalent to daily short-sale volume.

## Publication-time rule

The endpoint identifies records by `settlement_date`, but settlement-date
information is not public on that date. A record is usable only after FINRA's
official publication date for that settlement cycle.

For this January-March 2026 development probe, use the published FINRA schedule:

| Settlement date | Publication date |
|---|---|
| 2025-11-14 | 2025-11-25 |
| 2025-11-28 | 2025-12-09 |
| 2025-12-15 | 2025-12-24 |
| 2025-12-31 | 2026-01-12 |
| 2026-01-15 | 2026-01-27 |
| 2026-01-30 | 2026-02-10 |
| 2026-02-13 | 2026-02-25 |
| 2026-02-27 | 2026-03-10 |
| 2026-03-13 | 2026-03-24 |
| 2026-03-31 | 2026-04-10 |

For a candidate event on trading day D:

- a short-interest record is eligible only when its official publication date is
  strictly before D;
- a record published on D is not used because date-level publication metadata
  does not prove availability before the intraday decision;
- records with unknown settlement-to-publication mapping are ignored, never
  guessed;
- no future publication may be backfilled.

## Sampling

Reuse the same deterministic candidate-v2 signal construction as the previous
supply probes. Within each held-out development month, sort candidate signals by
trading day, threshold, ticker, and timestamp and select 25 approximately evenly
spaced events when at least 25 exist. Maximum sample: 75 events.

Selection must not use later realized returns.

## Probe metrics

For each sampled event report:

- number of publication-eligible short-interest records;
- official publication date of the latest eligible record;
- calendar age of that publication at the event date;
- latest settlement date;
- latest `short_interest`;
- latest `avg_daily_volume`;
- latest `days_to_cover`;
- whether two publication-eligible records exist;
- change in short interest between the latest two eligible reports when both are
  available.

Aggregate by month and overall.

Do not inspect realized returns or tune any threshold based on outcomes.

## Go / no-go rule

The source is sufficiently available for a later pre-registered model branch
only if:

1. at least 80% of sampled events have at least one publication-eligible record;
2. every development month individually has at least 70% such coverage; and
3. at least 60% of sampled events have two publication-eligible records, so a
   causal position-change feature is feasible.

If the probe fails, do not substitute settlement date for publication date or
relax the publication rule based on January-March outcomes.
