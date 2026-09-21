# Expanded-history point-in-time supply-state hurdle EV v2.7 — pre-registration

## Status

Pre-registered after v2.6 was completed and retired and after the independent
point-in-time share-supply availability probe passed, before implementation and
before viewing any v2.7 trading result.

Use only September 2025 through March 2026 development data. April 2026 and
later remain sealed.

## Motivation fixed before results

v2.6 applied a structural execution-feasibility universe before fitting the
existing hurdle decomposition. Broad out-of-month hurdle ordering remained
positive in January, February and March, but the selected EV > 0 tail was
gross-negative in all three months.

The v2.6 failure rule therefore identifies missing candidate/state information,
rather than threshold calibration or execution friction, as the next
development bottleneck.

A separately pre-registered supply probe then established that Massive
point-in-time ticker details queried only at D-1 calendar day provide:

- 100% request success in each sampled development month;
- 100% weighted-shares coverage;
- 92% / 92% / 96% share-class-shares coverage;
- 100% causally reconstructed prior market-cap coverage.

No realized-return relationship was used to choose a supply threshold.

## Frozen candidate universe

Use the exact v2.6 first-anchor execution-feasibility rule:

- first causal eligible anchor only;
- dollar_volume_5m >= 100,000 USD;
- transactions_5m >= 50;
- active_minute_fraction_15m >= 0.80;
- if the first anchor fails, skip the ticker-day;
- never fall through to a later state.

The gate is applied after chronological fit/calibration day assignment, exactly
as in v2.6.

## Point-in-time share-supply enrichment

For every execution-feasible first anchor on trading day D, query:

    GET /v3/reference/tickers/{ticker}?date=D-1 calendar day

Never query D or a later date. Never use current ticker details as fallback.
Non-gated anchors remain in the anchor checkpoint for fold/date provenance but
do not require a reference request.

Persist the raw point-in-time fields:

- supply_weighted_shares_outstanding_prior;
- supply_share_class_shares_outstanding_prior;
- supply_provider_market_cap_prior;
- supply_query_date;
- supply_query_success.

The provider market-cap field is diagnostic only and is not a model feature.

## Frozen new model features

Add exactly these continuous causal features to the existing v2.4 multi-source
feature frame:

1. supply_log_weighted_shares_prior
   = log(weighted_shares_outstanding_prior)
2. supply_log_share_class_shares_prior
   = log(share_class_shares_outstanding_prior)
3. supply_log_implied_market_cap_prior
   = log(weighted_shares_outstanding_prior * previous_close)
4. supply_weighted_share_turnover_5m
   = volume_5m / weighted_shares_outstanding_prior
5. supply_share_class_turnover_5m
   = volume_5m / share_class_shares_outstanding_prior
6. supply_market_cap_turnover_5m
   = dollar_volume_5m /
     (weighted_shares_outstanding_prior * previous_close)
7. supply_share_class_to_weighted_ratio
   = share_class_shares_outstanding_prior /
     weighted_shares_outstanding_prior

No low-float, market-cap, turnover, or share-count threshold may be selected
from January-March returns. Missing share-class values remain NaN and are
handled by the frozen tree model rather than causing candidate exclusion.

## Frozen folds

Evaluate only:

- January 2026 using Sep-Dec 2025 strictly-prior history;
- February 2026 using Sep 2025-Jan 2026 strictly-prior history;
- March 2026 using Sep 2025-Feb 2026 strictly-prior history.

Use the exact v2.4/v2.6 chronological day assignment:

- first 80% of strictly-prior scoreable trading days for fitting;
- final 20% for calibration.

Persist date provenance and prove calibration dates precede evaluation.

## Frozen models

Train two model sets on the same execution-feasible fit/calibration anchors.

### Baseline comparator

Exact v2.4 three-head hurdle model using the existing multi-source feature frame.

### Supply model — primary

Exact same model classes, hyperparameters, seeds, winsorization and calibration
rules as v2.4, except the seven pre-registered supply features above are appended
to the input frame for all three heads.

Heads:

1. calibrated P(15m BASE return > 0);
2. conditional positive BASE-return magnitude;
3. conditional non-positive BASE-loss magnitude.

Compute:

    hurdle_EV =
        P(win) * E[gain | win, X]
        - (1 - P(win)) * E[loss | loss, X]

No final affine scale calibration is allowed.

## Executable policies

All policies use the same execution-feasible first anchor, fixed 15-minute BUY,
and at most one attempt per ticker-day.

### feasible_earliest_15m_cap1

BUY every execution-feasible anchor.

### feasible_base_hurdle_ev_15m_cap1

BUY iff the baseline comparator hurdle_EV > 0.

### supply_hurdle_ev_15m_cap1 — primary

BUY iff the supply-augmented hurdle_EV > 0.

Zero is a semantic expected-value boundary and must not be tuned.

No timing rank, HOLD, SELL, later-anchor selection or horizon freedom is
allowed.

## Required diagnostics

For each evaluation month report:

- fit/calibration/evaluation date provenance;
- pre-gate and post-gate anchor counts;
- supply request and field coverage;
- baseline and supply probability AUC;
- baseline and supply conditional win/loss magnitude Spearman;
- baseline and supply hurdle-EV Spearman versus realized BASE;
- supply selected predicted EV versus realized gross/BASE/stress;
- primary attempts/evaluated/unevaluable reasons;
- BASE mean, day-balanced BASE, p05, severe-loss rate and worst day;
- direct primary-vs-baseline selected-policy comparison;
- pooled trading-day cluster bootstrap;
- frozen position-ledger light/base/stress marked returns and accounting
  completeness.

## Development promotion rule

Do not access April 2026 or later unless ALL conditions hold for
supply_hurdle_ev_15m_cap1:

1. at least 15 evaluated primary trades in January, February and March;
2. arithmetic BASE mean > 0 in every month;
3. day-balanced BASE mean > 0 in every month;
4. day-balanced BASE mean strictly exceeds feasible_earliest_15m_cap1 in every
   month;
5. monthly BASE mean is not worse than feasible_base_hurdle_ev_15m_cap1 in any
   month;
6. BASE p05 and STRESS mean are not worse than feasible_earliest_15m_cap1 in any
   month;
7. supply hurdle-EV Spearman versus realized BASE > 0 in every month;
8. pooled day-cluster bootstrap 95% lower bound > 0;
9. weighted-share and implied-market-cap coverage remain >= 80% in every month;
10. BASE position-ledger marked return > 0 and complete_accounting=true in every
    month.

A sparse tail, zero-trade month, light-only profit or missing outcome cannot
satisfy promotion.

## Failure rule

If v2.7 fails, do not tune supply thresholds, feature subsets, hurdle zero
crossing, model capacity, costs, horizon or winsorization on January-March.

Interpret failure as follows:

- if supply hurdle ordering is not consistently positive, retire this exact
  supply representation;
- if ordering improves but the selected tail remains gross-negative, the model
  still lacks causal continuation/exhaustion information and the next branch
  must add a genuinely different state source rather than more calibration;
- if selected gross alpha is repeatedly positive but BASE remains negative,
  execution-cost observability becomes the next bottleneck;
- only after a replicating cost-positive entry process exists may timing, HOLD
  or SELL freedom be introduced.

No fresh validation month is consumed here.
