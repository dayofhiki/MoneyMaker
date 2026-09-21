# Point-in-time split-history probe v0.1 — pre-registration

## Status

Pre-registered after v2.7 was completed and retired, before implementation and before viewing any split-history relationship with returns.

Use only January-March 2026 development anchors. April 2026 and later remain sealed.

## Motivation

v2.7 added point-in-time share-supply variables. Those variables preserved positive broad hurdle ordering but the selected tail remained gross-negative in all three months, with March deteriorating sharply.

The v2.7 failure rule therefore requires a genuinely different causal state source rather than more supply thresholds, score calibration, or cost tuning.

Recent corporate-action history is structurally different from the existing price/volume/breadth/short/8-K/share-count state. Reverse splits in particular can mark a different corporate lifecycle and listing-maintenance regime that may matter for small-cap continuation/exhaustion.

This probe tests availability only. It must not inspect or condition on realized returns.

## Frozen sample

Source artifact: exact completed v2.7 development run 74.

Use policy attempts from `feasible_earliest_15m_cap1`, not only evaluated trades, so sampling does not depend on future label availability.

For each of January, February and March 2026:

1. sort attempts by trading_day, ticker, decision_t;
2. drop duplicate ticker-days;
3. deterministically choose exactly 25 anchors by evenly spaced indices.

Target sample size: 75 anchors total.

## Point-in-time query

For an anchor on trading day D and ticker T, query Massive:

    GET /stocks/v1/splits

with:

- ticker = T;
- execution_date.gte = D - 730 calendar days;
- execution_date.lte = D - 1 calendar day;
- sort = execution_date.asc;
- limit = 1000.

Never include D or a later execution date.

The new Massive splits endpoint is used, not the deprecated /v3/reference/splits endpoint.

Persist every returned record's:

- ticker;
- execution_date;
- split_from;
- split_to;
- adjustment_type;
- historical_adjustment_factor when present.

An empty result is a valid successful request meaning no split record in the fixed lookback.

## Availability summaries

For each sampled anchor derive, without using any return:

- request_success;
- total split count in 730d;
- reverse split count in 730d;
- reverse split count in 365d;
- forward split count in 730d;
- stock-dividend count in 730d;
- days since latest split;
- days since latest reverse split;
- latest reverse split ratio = split_to / split_from;
- maximum reverse consolidation factor = split_from / split_to among reverse splits.

## Go / no-go rule

Proceed to a separately pre-registered return-model branch only if ALL hold:

1. request success >= 95% in every month;
2. every persisted execution_date is strictly before its anchor trading day;
3. at least 10 of the 75 sampled anchors have a reverse split in the prior 730 days;
4. every month has at least 2 sampled anchors with a reverse split in the prior 730 days.

These support thresholds are frozen before observing probe results. They do not use realized returns.

If the probe fails, do not lower the prevalence threshold on January-March. Retire split history as the next primary state source and test a different independent source.

If the probe passes, pre-register exactly one v2.8 branch that adds continuous/categorical split-history features to the frozen v2.7 execution-feasible hurdle setup. No split threshold may be selected from January-March outcomes.

No fresh validation month is consumed by this probe.
