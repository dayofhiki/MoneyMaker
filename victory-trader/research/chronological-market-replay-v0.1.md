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

The same rule applies to duplicate minute keys. A conflicting ticker is
replaced by its exact-date, unadjusted REST minute series only after every
conflicting timestamp matches exactly one Flat File candidate on OHLCV.

Missing WATCH/HOT names retain their observation tier during the configured
grace window and therefore reserve capacity until they return or DROP. This
keeps the configured observation budgets strict even when the broad scan has
intermittent bars.

## Phase 2B smoke result

Request 107 completed the 2026-01-02 market-wide smoke replay. Artifact audit
verified 482,710 scan rows, 1,054,089 trace rows, 2,748 scan symbols and 390
regular-session timestamps, with zero duplicate keys, no missing attention
scores and chronological ordering in both Parquet outputs. The adapter
resolved two conflicting prior-close tickers and two conflicting minute-series
tickers against uniquely matching exact-date REST data.

The first artifact exposed a capacity bug during intermittent bars: missing
grace records were retained without reserving their WATCH/HOT slots. After the
runtime fix, request 107 held maximum occupancy to exactly WATCH 50 and HOT 10.
Its +10% runner coverage was 53.44% at WATCH-or-better and 3.44% at
HOT-or-position, with a five-minute median WATCH lead. These figures are an
infrastructure baseline only and must not be used to tune v0.1 thresholds.

## Phase 2C — completed

Request 108 completed the unchanged 2026-01-02 through 2026-01-05 replay across
two regular sessions. The artifact contains 990,188 scan rows and 2,113,347
trace rows over 780 regular-session timestamps. Both sessions preserved strict
WATCH 50 / HOT 10 occupancy limits, chronological ordering, session reset and
per-day duplicate resolution. The combined frozen infrastructure baseline
captured 316 of 655 eventual +10% close-based crossings at WATCH-or-better
(48.24%) and 21 at HOT/POSITION (3.21%). Median WATCH lead was one minute.
These remain coverage/resource diagnostics only.

The two-session artifact also exposed the next scaling constraint: concatenating
all scan and trace frames in memory is appropriate for smoke tests but not for a
month-scale market-wide replay.

## Phase 2D — day-partitioned streaming replay

`victory_trader.attention_partitioned_replay` writes each completed trading
session immediately to separate scan and trace Parquet partitions, while
retaining only one session in memory. Its incremental summary preserves the
same exact replay counts, capture rates, occupancy maxima, unique-symbol count
and runner lead-time medians as the monolithic implementation.

A parity test replays two sessions through both implementations and requires the
partition contents and replay summary to match exactly. Request 109 reran the
same frozen 2026-01-02 through 2026-01-05 slice through the partitioned path
and completed successfully. Its aggregate replay diagnostics exactly matched
request 108: 2,113,347 trace rows, 655 runner episodes, 48.24% WATCH-or-better
capture, 3.21% HOT/POSITION capture, WATCH 50 / HOT 10 peak occupancy, and one
minute median WATCH lead. The output is now split into one scan and one trace
Parquet partition per trading day.

## Phase 2E — first multi-session development expansion

Request 110 computed the unchanged partitioned replay through 2026-01-09,
covering the first six January trading sessions. The replay itself completed
successfully, but the GitHub run is marked failed because final artifact
finalization returned HTTP 403 after uploading roughly 99 MB of full scan/trace
partitions. This is an artifact-retention failure, not a replay-computation
failure.

The completed request-110 summary covered 6,300,821 trace rows and 1,443
eventual +10% runner crossings. WATCH-or-better captured 972 crossings
(67.36%), while HOT/POSITION captured 63 (4.37%). Median WATCH lead was three
minutes and median HOT lead remained zero minutes. WATCH/HOT occupancy again
reached the frozen 50/10 limits.

Per-session WATCH-or-better capture varied materially: 53.44%, 43.28%, 89.55%,
83.50%, 66.83% and 94.89% from January 2 through January 9 trading sessions.
HOT/POSITION capture remained only 2.99% to 6.25%, with zero-minute median HOT
lead on every session. These observations motivate new causal attention
information rather than tuning the frozen score thresholds or capacities.

To keep longer development runs durable, the workflow now compacts completed
partitions before upload. It retains WATCH/HOT/POSITION focus rows, runner
crossing rows, the exact replay summary and the log, while the full per-day
partitions remain ephemeral workflow working data. Request 112 repeats the same
six-session frozen computation only to verify compact artifact retention; it
does not change any research rule.

## Phase 2E compact artifact validation — request 112

Request 112 successfully reran the unchanged six-session 2026-01-02 through
2026-01-09 replay and preserved the exact request-110 aggregate diagnostics.

The compact retention path reduced the durable artifact to 758,513 bytes while
preserving 140,400 WATCH/HOT/POSITION focus rows, all 1,443 runner crossing
rows, the complete summary and execution log. This removes artifact size as the
immediate scaling bottleneck without changing replay semantics.

The six-session baseline remains:

- 6,300,821 total trace rows;
- 1,443 +10% close-based runner crossings;
- 972 WATCH-or-better captures, 67.36%;
- 63 HOT/POSITION captures, 4.37%;
- median WATCH lead 3 minutes;
- median HOT lead 0 minutes;
- WATCH/HOT occupancy maxima 50/10.

Because the current score is only same-timestamp cross-sectional return rank,
these figures are an infrastructure baseline rather than a validated attention
model. The next research problem is improving causal WATCH-to-HOT prioritization
with genuinely new intraday information rather than tuning these frozen
thresholds.


## Phase 2F — causal minute-completion timestamp correction

A timestamp audit before second-level enrichment found that Massive minute
aggregate timestamps are bar-start timestamps, while the v0.1 scanner consumes
that bar's close, volume and transaction count. Those values are not fully
observable at the bar start.

The scan builder now preserves the provider timestamp as `bar_start_t` and
sets the replay decision timestamp to:

`decision_t = bar_start_t + 60,000 ms`.

This is a causality correction, not a score, threshold, capacity, universe or
outcome change. Because every completed-minute observation and its +10% crossing
audit move by the same 60 seconds, the existing state sequence and relative
lead-time arithmetic should remain unchanged. Request 114 repeats the frozen
six-session January slice to verify that expectation before any second-level
feature experiment is allowed to proceed.
