# Expanded-history split-state hurdle EV v2.8 — pre-registration

## Status

Pre-registered after v2.7 was completed and retired and after the independent
split-history availability probe passed, before implementation and before
viewing any v2.8 return result.

Use only September 2025 through March 2026 development data. April 2026 and
later remain sealed.

## Motivation fixed before results

The frozen v2.7 supply-state hurdle model preserved positive broad hurdle
ordering but its selected EV>0 tail was gross-negative in January, February,
and March. The v2.7 failure rule therefore requires a genuinely different
causal state source rather than additional supply thresholds, calibration,
cost tuning, or policy freedom.

The independent split-history probe then established, without using realized
returns, that strictly-prior corporate-action history is available and common
enough to learn from: 18 of 75 sampled development anchors had a reverse split
within the prior 730 days, with support in every month.

This branch tests exactly one intervention: append frozen split-history state to
the existing v2.7 supply-state representation.

## Frozen candidate universe

Use the exact v2.6/v2.7 first-anchor execution-feasibility rule:

- first causal eligible anchor only;
- dollar_volume_5m >= 100,000 USD;
- transactions_5m >= 50;
- active_minute_fraction_15m >= 0.80;
- if the first anchor fails, skip the ticker-day;
- never fall through to a later anchor.

## Frozen split-history enrichment

For each first anchor on trading day D and ticker T, use only split records with:

- ticker = T;
- execution_date >= D - 730 calendar days;
- execution_date <= D - 1 calendar day.

Never use D or later records.

For operational efficiency, implementation may fetch the market-wide split
history once over the global required calendar interval and perform the exact
ticker/date filtering locally. This retrieval optimization does not alter
feature semantics.

Persist raw/diagnostic support fields and derive exactly these eight model
features:

1. split_any_730d
   = 1 if any split record exists in the prior 730 days, else 0;
2. reverse_split_count_730d;
3. reverse_split_count_365d;
4. forward_split_count_730d;
5. stock_dividend_count_730d;
6. split_log1p_days_since_latest_reverse
   = log1p(days since latest reverse split), missing if none;
7. split_log_latest_reverse_consolidation
   = log(split_from / split_to) for the latest reverse split, missing if none;
8. split_log_max_reverse_consolidation
   = log(max split_from / split_to among reverse splits in the window),
     missing if none.

No low/high split-count threshold, reverse-split subgroup, age cutoff,
consolidation cutoff, or interaction may be selected from January-March
returns.

## Frozen folds

Evaluate only:

- January 2026 using Sep-Dec 2025 strictly-prior history;
- February 2026 using Sep 2025-Jan 2026 strictly-prior history;
- March 2026 using Sep 2025-Feb 2026 strictly-prior history.

Use the exact v2.7 chronological fit/calibration split and persist provenance
proving all calibration dates precede evaluation.

## Frozen models

Train two model sets on the same execution-feasible anchors.

### Supply comparator

Exact v2.7 supply-state hurdle model, unchanged.

### Supply + split model — primary

Use the exact v2.7 model classes, hyperparameters, seeds, target
winsorization/calibration rules and seven supply features, then append only the
eight pre-registered split-history features above.

Heads remain:

1. calibrated P(15m BASE return > 0);
2. conditional positive BASE-return magnitude;
3. conditional non-positive BASE-loss magnitude.

Compute the exact hurdle expected value:

    hurdle_EV =
        P(win) * E[gain | win, X]
        - (1 - P(win)) * E[loss | loss, X]

No final affine scale calibration is allowed.

## Executable policies

All policies use the same execution-feasible first anchor, fixed 15-minute BUY,
and at most one attempt per ticker-day.

### feasible_earliest_15m_cap1

BUY every execution-feasible anchor.

### supply_hurdle_ev_15m_cap1

Exact v2.7 supply comparator. BUY iff supply hurdle_EV > 0.

### split_supply_hurdle_ev_15m_cap1 — primary

BUY iff supply+split hurdle_EV > 0.

Zero remains a semantic expected-value boundary and must not be tuned.

No timing rank, HOLD, SELL, later-anchor selection, alternative horizon, or
position-sizing freedom is allowed.

## Required diagnostics

For each evaluation month report:

- fit/calibration/evaluation date provenance;
- pre/post execution-feasibility anchor counts;
- split-history strict-prior and support coverage;
- supply comparator and split-supply probability AUC;
- conditional win/loss magnitude Spearman for both;
- hurdle-EV Spearman versus realized BASE for both;
- primary selected predicted EV and realized gross/BASE/stress;
- primary attempts/evaluated/unevaluable reasons;
- BASE mean, day-balanced BASE, p05, severe-loss rate and worst day;
- direct primary-vs-supply-comparator and primary-vs-earliest comparisons;
- pooled trading-day-cluster bootstrap;
- frozen position-ledger light/base/stress marked returns and accounting
  completeness.

## Development promotion rule

Do not access April 2026 or later unless ALL conditions hold for
split_supply_hurdle_ev_15m_cap1:

1. at least 15 evaluated primary trades in January, February and March;
2. arithmetic BASE mean > 0 in every month;
3. day-balanced BASE mean > 0 in every month;
4. day-balanced BASE mean strictly exceeds feasible_earliest_15m_cap1 in every
   month;
5. monthly BASE mean is not worse than supply_hurdle_ev_15m_cap1 in any month;
6. BASE p05 and STRESS mean are not worse than feasible_earliest_15m_cap1 in
   any month;
7. split-supply hurdle-EV Spearman versus realized BASE > 0 in every month;
8. pooled day-cluster bootstrap 95% lower bound > 0;
9. split query/enrichment is strictly prior for every populated record and
   reverse-split support remains nonzero in every evaluation month;
10. BASE position-ledger marked return > 0 and complete_accounting=true in every
    month.

A sparse tail, zero-trade month, light-only profit, or missing outcome cannot
satisfy promotion.

## Failure rule

If v2.8 fails, do not tune split thresholds, feature subsets, hurdle zero
crossing, model capacity, costs, horizon, winsorization, or corporate-action
lookback on January-March.

Interpret failure as follows:

- if split-supply hurdle ordering is not consistently positive, retire this
  split representation;
- if ordering remains positive but the selected tail is gross-negative, the
  model still lacks continuation/exhaustion information and the next branch
  must probe another genuinely different causal source;
- if selected gross alpha is repeatedly positive but BASE remains negative,
  execution-cost observability becomes the next bottleneck;
- only after a replicating cost-positive entry process exists may timing,
  HOLD/SELL, horizon, or sizing freedom be introduced.

No fresh validation month is consumed here.


## Result — request 77

Request 76 failed in tests because the synthetic fixture omitted full upstream
anchor inputs; no model result was produced. Request 77 changed only that
fixture and completed successfully with the frozen v2.8 design. April 2026 and
later remained sealed.

| Month | Evaluated trades | Gross mean | BASE mean | Day-balanced BASE | Bootstrap 95% low |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 47 | +1.007% | -0.036% | +0.289% | -1.475% |
| 2026-02 | 10 | -1.297% | -2.741% | -2.377% | -5.240% |
| 2026-03 | 4 | -5.114% | -6.058% | -6.058% | -11.350% |

January showed a useful gross improvement, but it did not cover BASE friction,
its bootstrap lower bound was negative, and its BASE ledger marked return was
-1.151%. The effect did not replicate: February and March were gross-negative,
sparse, and BASE-negative. March selected the exact same four evaluated trades
as the v2.7 supply comparator.

## Decision

Reject exact v2.8. Do not access April 2026 or later. Do not tune split
thresholds, feature subsets, lookback, EV boundary, costs, model capacity, or
horizon on January-March. Split history did not provide a stable
continuation-versus-exhaustion distinction. The next branch must use a genuinely
different strictly-prior state source.
