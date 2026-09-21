# Point-in-time share-supply probe v0.1 — pre-registration

## Status

Pre-registered after expanded-history execution-feasible hurdle EV v2.6 was
completed and retired, before implementation and before viewing any result from
this probe.

This is a data-supply probe only. It uses January-March 2026 development
anchors already consumed during model development. April 2026 and later remain
sealed.

## Motivation

v2.6 preserved positive hurdle-score ordering after a structural liquidity gate
but its selected tail was gross-negative in every development month. The
pre-registered v2.6 failure rule therefore directs the next intervention toward
genuinely new causal state information.

Current state features cover intraday price/volume, market breadth, FINRA short
volume/interest and coarse prior 8-K categories. They do not contain the
company's point-in-time share supply, public float or market capitalization.

For small-cap momentum, a fixed amount of observed buying pressure can have very
different economic meaning when the tradable share supply differs by orders of
magnitude. This probe asks only whether point-in-time reference share data is
available with sufficient coverage to justify a separately pre-registered model
branch.

## Source

Use the existing Massive client endpoint:

    GET /v3/reference/tickers/{ticker}?date=YYYY-MM-DD

The endpoint supports point-in-time ticker details and may return:

- share_class_shares_outstanding;
- weighted_shares_outstanding;
- market_cap.

No current-date fallback is allowed.

## Sampling

Input is the completed v2.6 Jan-Mar development trade artifact.

For each month:

1. keep only feasible_earliest_15m_cap1 rows;
2. sort by trading_day, ticker, t;
3. remove duplicate ticker-days;
4. choose 25 approximately evenly spaced rows when at least 25 exist.

Maximum sample is 75 ticker-days.

Sampling must not use realized returns, hurdle predictions or future labels.

## Point-in-time rule

For an anchor on trading day D, request ticker details with date D-1 calendar
day.

The query date is therefore strictly earlier than the intraday decision day,
including Monday/weekend cases. Never query D or a later date, never fall back
to current details, and never backfill a missing historical field from a later
response.

For each sampled row compute only diagnostics:

- whether the point-in-time request succeeded;
- share-class shares outstanding;
- weighted shares outstanding;
- provider market cap;
- implied market cap = weighted shares outstanding * previous_close;
- five-minute share turnover = volume_5m / share-class shares outstanding;
- five-minute dollar turnover = dollar_volume_5m / implied market cap.

The two turnover ratios are diagnostic only in this probe.

## Go / no-go rule

The source is sufficiently available to justify a model branch only if ALL hold:

1. ticker-details request success >= 95% in every month;
2. weighted-shares coverage >= 80% in every month;
3. share-class-shares coverage >= 80% in every month;
4. implied-market-cap coverage >= 80% in every month;
5. every recorded query date is strictly earlier than its trading day.

These are availability and causality criteria only, not profitability criteria.

If the probe passes, pre-register a separate supply-state model branch before
looking at that branch's trading performance. The branch may use the causal
share-supply fields and fixed ratios above, but must not tune thresholds from
January-March outcomes.

If the probe fails, do not lower the coverage criteria or substitute present-day
reference values based on these results.


## Result — request 72

Workflow run 35566495316 completed successfully with 75 deterministic
development anchors, 25 from each of January-March 2026.

The probe made 75 point-in-time ticker-details network requests with zero
retries and no current-date fallback.

| Month | Request success | Weighted shares | Share-class shares | Provider market cap | Implied market cap |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 100% | 100% | 92% | 100% | 100% |
| 2026-02 | 100% | 100% | 92% | 100% | 100% |
| 2026-03 | 100% | 100% | 96% | 100% | 100% |

Every query used D-1 calendar day and therefore satisfied the strict-prior date
check.

All pre-registered supply criteria passed.

The sampled supply ratios also spanned orders of magnitude, confirming that the
source adds a materially different state axis rather than a near-constant
identifier. Examples ranged from sub-0.1% five-minute market-cap turnover to
multi-100% turnover in very small-share-count names.

## Decision

Proceed to a separately pre-registered development branch that augments the
v2.6 execution-feasible first-anchor universe with point-in-time share-supply
features.

No January-March return outcome was used to choose a float/share threshold. The
new branch will feed continuous causal supply variables into the model rather
than hand-selecting a low-float subgroup.
