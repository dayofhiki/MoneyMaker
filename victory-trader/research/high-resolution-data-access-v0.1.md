# High-resolution data access probe v0.1

## Status

Pre-registered infrastructure probe. Read-only and development-only.

No subscription, entitlement, brokerage, threshold, policy, or validation-period
change is allowed by this probe. April 2026 and later remain sealed.

## Purpose

The hierarchical trader now has an explicit transition from broad minute-level
SCAN/WATCH monitoring to higher-resolution HOT/POSITION observation. Before
building the next information-value experiment, establish which historical
high-resolution sources are actually available under the repository's existing
Massive credentials.

Previous request 51/52 work established that historical NBBO REST and quote Flat
Files returned HTTP 403 under the then-current entitlement. This probe does not
assume that trade or one-second aggregate access has the same entitlement.

## Frozen probe

Use:

- ticker: `AAPL`
- trading day: `2026-01-02`

The ticker is only an entitlement/data-path probe. It is not a research
candidate and no price outcome is evaluated.

Check, without changing any plan:

1. REST one-second aggregate bars for 2026-01-02;
2. REST tick trades for the first 10 seconds of the regular session;
3. S3 HEAD access to `us_stocks_sip/trades_v1`;
4. S3 HEAD access to `us_stocks_sip/quotes_v1`.

The Flat File checks must not download full trade/quote files.

## Interpretation

This probe answers only which historical inputs can be used in the next causal
research layer.

- one-second aggregates available: HOT names can be studied at second
  resolution immediately;
- trades available: tick-level price/size/event features can be studied for
  selected HOT/POSITION names;
- quotes available: NBBO spread/depth features and empirical top-of-book
  execution diagnostics can be revisited;
- unavailable sources remain unavailable. Do not impute them or change the
  subscription automatically.

No result from this probe is a profitability claim.

## Result — request 113

The read-only retry completed successfully after fixing only ticker
normalization in the probe test.

- REST one-second aggregates: **AVAILABLE**.
  - AAPL on 2026-01-02 returned 18,379 one-second aggregate rows.
- REST historical trades: **UNAVAILABLE**, HTTP 403.
- trade Flat File HEAD: **UNAVAILABLE**, HTTP 403.
- quote Flat File HEAD: **UNAVAILABLE**, HTTP 403.

The earlier historical NBBO REST probe also returned HTTP 403. Therefore the
next high-resolution research layer should use one-second aggregate bars under
the existing entitlement and must not assume tick trades or NBBO are available.

This is an information-access result only, not evidence of predictive edge or
profitability.
