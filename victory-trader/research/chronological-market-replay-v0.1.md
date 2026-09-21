# Chronological market replay v0.1

## Status

Phase 2A implemented after hierarchical attention runtime v0.1 and before any
market-return-based scanner tuning.

April 2026 and later remain sealed. This implementation does not open a new
month, fit an entry model, or claim improved profitability.

## Purpose

The attention runtime needs to be evaluated as a sequential market process,
not as independent rows. The replay layer now:

1. constructs a low-cost market-wide scan frame from chronological minute bars
   and point-in-time prior closes;
2. advances a fresh attention runtime through every session timestamp;
3. records every state, resolution request, rank and transition reason;
4. audits whether eventual +10% runners were already WATCH or HOT by their
   first close-based crossing;
5. reports observation-tier demand without using trade returns.

## Frozen infrastructure baseline

The v0.1 `attention_score` is the same-timestamp cross-sectional percentile of
nominal return from the prior close.

This baseline is intentionally simple:

- it is available at the decision timestamp;
- it compares the current market rather than relying on a static ticker list;
- it uses no future return, completed-day high, completed-day volume or
  hindsight-best horizon;
- it gives the finite WATCH/HOT allocator a deterministic input;
- it is not presented as the final attention model.

The research universe remains nominal prior close $0.50–$20 plus an explicit
point-in-time eligibility flag. Phase 2B must populate that flag from the
existing historical common-stock/exchange taxonomy rather than infer it from
future metadata.

## Audit outputs

The replay summary contains:

- sessions and unique symbols;
- runner episodes;
- WATCH-or-better capture by the first +10% crossing;
- HOT/POSITION capture by the first +10% crossing;
- median WATCH and HOT lead time;
- maximum WATCH and HOT occupancy;
- state-transition count;
- grouped-minute, minute-bar, second-bar and trades/NBBO request counts;
- DROP rows caused by invalid/ineligible/missing observations.

These are coverage and resource diagnostics. They must not be described as
economic performance.

## Leakage guard

`replay_attention` consumes an explicit allowlist: trading day, ticker,
timestamp, attention score, and optional current position/data-quality flags.
Outcome columns are ignored. The tests inject adversarial future-return values
and require the full trace to remain identical.

Sessions use fresh attention state so prior-day HOT hysteresis cannot leak into
the next opening scan. Open-position carry is not supported in this phase; the
current research convention is regular-session liquidation.

## Phase 2B — implemented, smoke replay requested

Add a Massive Flat Files adapter that:

1. loads the complete minute aggregate file for each selected January–March
   development day;
2. joins nominal prior-session closes;
3. joins point-in-time common-stock eligibility and excludes same-day splits;
4. restricts inputs to the regular session;
5. builds the scan frame and runs this replay unchanged;
6. uploads the trace, summary and data-demand audit as a GitHub Actions
   artifact.

The first run should be a bounded January development slice. Capacity settings
and score thresholds remain fixed before its runner-capture results are
inspected. April 2026 and later remain sealed.

`victory_trader.attention_flatfile_replay` now implements this adapter and the
`attention_market_replay_v01` Actions operation. Dedicated request 103 freezes the first
smoke slice to 2026-01-02 with default runtime settings. Its purpose is to
verify end-to-end market coverage, taxonomy, split handling, chronological
state evolution and artifact production. The one-day result must not be used
to change score thresholds or claim profitability.

The adapter collapses duplicate prior-close rows when the normalized ticker
and close agree. For conflicting candidates it queries the same provider's
exact-date, unadjusted REST daily bar and accepts a candidate only when exactly
one value matches; ambiguous or inconsistent cases fail. The workflow also
propagates replay failures through `tee` and verifies all core outputs are
non-empty before artifact upload.
