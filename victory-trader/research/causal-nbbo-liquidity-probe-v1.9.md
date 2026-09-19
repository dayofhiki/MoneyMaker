# Causal NBBO liquidity / execution probe v1.9 — pre-registration

## Status

Frozen before quote outcomes. Development diagnostics only. Uses only the
already-seen January-March 2026 request-48 BUY-attempt ledger. No new month,
policy tuning, subscription change, brokerage action, or profitability claim.

## Purpose

The post-v1.8 audits show two unresolved bottlenecks:

1. frozen selected gross alpha is small relative to the synthetic fixed cost
   scenario; and
2. bar-reference evaluation omits or delays a material number of entries/exits.

Before fitting another policy, test whether historical top-of-book NBBO data is
actually available at the required timestamps and whether it can support
strictly causal liquidity features plus a more observable execution audit.

## Frozen sample

Use request 48 artifact `state-calibrated-sequential-q-v18-attempts.csv`.

- policy: `raw_same_fit_cap1` only;
- 20 attempts per month, or all attempts if fewer;
- sort by trading_day, decision_t, ticker and take evenly spaced deterministic
  indices;
- sampling must not inspect any return/outcome column.

This yields at most 60 probes.

## Timestamp semantics

The state row timestamp is the bar start. The decision bar is fully known at:

`decision_known_t = decision_t + 60,000 ms`.

The fixed 10-minute position geometry implies:

- entry target: `decision_known_t`;
- scheduled exit target: `decision_t + 11 * 60,000 ms`.

These timestamps are accounting/research targets, not claims of zero-latency
broker fills.

## Strictly causal decision quote

Query only `[decision_known_t - 5s, decision_known_t]`.

Choose the latest quote satisfying:

- SIP timestamp <= decision_known_t;
- bid and ask finite and > 0;
- ask >= bid;
- bid_size and ask_size finite and > 0.

No after-target fallback is permitted.

Record quote age, spread %, bid/ask sizes, displayed-dollar depth, size
imbalance, and microprice displacement from mid.

A decision quote is called `fresh_2s` only if age <= 2,000 ms. Quotes older
than 2s remain diagnostics but are not eligible future model features.

## Execution-observation quotes

These are outcomes/diagnostics and can never be features.

At entry and exit target separately, query the first valid quote with SIP
timestamp in `[target, target + 1s]`. Record:

- ask at entry and bid at exit;
- quote delay from target;
- displayed dollar depth on the marketable side;
- whether displayed top-of-book depth covers the frozen $1,000 audit notional.

If both are observed, calculate top-of-book ask-to-bid return. It includes
crossing the displayed spread but excludes commissions, additional slippage,
queue effects, partial fills and market impact. It is not an executable-fill
claim.

## Feasibility criteria

This probe is not a trading-model promotion test.

Proceed to an NBBO-enriched development model only if:

1. no 401/403 historical-quote authorization failure;
2. decision `fresh_2s` coverage >= 80% in every month;
3. entry and exit observation coverage >= 80% in every month;
4. at least 70% of observations with quotes have displayed top-of-book depth
   sufficient for the frozen $1,000 notional on both entry ask and exit bid.

If a criterion fails, do not impute quotes or loosen quote age/window after
inspection. Retain the bar-based research and choose a different information
source or development protocol.

## Reporting

Report monthly and pooled:

- authorization status and request counts;
- decision quote coverage/freshness;
- spread median/p75/p90;
- bid/ask size and imbalance;
- entry/exit observation coverage and delay;
- top-of-book depth sufficiency;
- top-of-book ask-to-bid return versus the existing bar gross/base return for
  the same sampled attempts where comparable;
- counts by missing reason.

No fresh validation month is consumed.


## Result — request 51

The reporting-only retry completed successfully and established the data-access
result without changing any preregistered probe rule.

- Historical `/v3/quotes/{ticker}` request: HTTP 403.
- Network requests before stop: 1.
- No quote observation was admitted.
- All four feasibility criteria failed because historical quote authorization
  is unavailable under the currently configured REST entitlement.
- No quote coverage, spread, depth or return statistic was inferred from the
  authorization failure.

Request 50 had already encountered the same empty-result path but crashed while
formatting an empty summary. Request 51 changed only that reporting path and is
the authoritative result.

## Decision

Do not impute NBBO, use after-target quotes as causal features, relax quote-age
rules, or purchase/change a data plan automatically.

Check existing quote Flat File entitlement read-only. If it is also unavailable,
retire the NBBO-enrichment branch under current entitlements and move to a
different causal information source / broader development protocol.

No fresh validation month was consumed.
