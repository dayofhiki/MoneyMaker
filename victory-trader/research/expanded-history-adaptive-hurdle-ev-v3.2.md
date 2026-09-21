# Expanded-history adaptive multi-horizon hurdle EV v3.2 — pre-registration

## Status

Pre-registered after v3.1 request 82 completed and before inspecting any v3.2
model output. Use only September 2025 through March 2026 development data.
April 2026 and later remain sealed.

## Motivation fixed before results

v3.1 showed that fixed 15-minute holding time is a genuine but incomplete
bottleneck.

For the v3.0 semantic selected tail:

- January had 48 trades, 37.5% horizon misses and 25.0% candidate misses; 75.0%
  had at least one BASE-positive frozen horizon and the hindsight best-horizon
  BASE mean was +2.881%.
- February had 6 trades, 0.0% horizon misses and 66.7% candidate misses; the
  hindsight best-horizon BASE mean remained -1.052%.
- March had 4 trades, 25.0% horizon misses and 75.0% candidate misses; the
  hindsight best-horizon BASE mean remained -2.513%.

Thus simply replacing 15 minutes with another fixed duration cannot solve the
cross-month failure. The next model must decide both whether any available
action has positive expected value and, if so, which holding horizon to choose.

## Frozen data construction

Start from the exact v3.0 filing-semantic anchors from request 79.

Reattach only the already-defined frozen state-panel outcome columns for:

- 1 minute;
- 2 minutes;
- 5 minutes;
- 10 minutes;
- 15 minutes;
- 30 minutes.

Sources are the original frozen state panels used by the research program:

- September-December 2025 from Historical State Development v2.0 request 54;
- January-March 2026 from State Trader Research v0.1 request 8.

Join only on trading_day, ticker and decision timestamp t. Existing 15-minute
labels must match the original state-panel labels exactly where both are finite.
A mismatch or unmatched anchor is an implementation failure.

These future-return columns are labels/evaluation only and never model features.

## Frozen causal feature set

Use the exact v3.0 filing-semantic + split + supply + multi-source causal feature
frame. No new feature, feature subset, interaction, threshold or external source
is selected in v3.2.

The execution-feasible first-anchor universe and chronological fit/calibration
split remain exactly those used in v2.6-v3.0.

## Action-value model

For each frozen horizon independently fit the exact v3.0 three-head hurdle
decomposition:

1. calibrated P(BASE return > 0);
2. conditional positive BASE-return magnitude;
3. conditional non-positive BASE-loss magnitude.

Use the exact v3.0 model classes, hyperparameters, winsorization, chronological
Platt calibration, magnitude offset calibration and random seeds. The 15-minute
head must therefore reproduce the v3.0 semantic 15-minute model apart from
reporting precision.

For horizon h:

hurdle_EV_h =
P(win_h) * E(win_magnitude_h)
- (1 - P(win_h)) * E(loss_magnitude_h).

## Frozen policies

### feasible_earliest_15m_cap1

Diagnostic broad comparator. Buy every execution-feasible first anchor for 15
minutes.

### filing_semantic_split_supply_hurdle_ev_15m_cap1

Exact-model comparator. Buy for 15 minutes iff the reproduced 15-minute semantic
hurdle EV is > 0.

### adaptive_filing_semantic_hurdle_ev_cap1 — primary

At the first execution-feasible anchor:

1. score all six horizon-specific hurdle EVs;
2. choose the horizon with the largest predicted hurdle EV;
3. BUY iff that maximum predicted EV is > 0;
4. otherwise SKIP.

The chosen action is committed before seeing label availability. If its entry or
future outcome is unevaluable, the first attempt is consumed. There is no
fallback to another horizon.

This is adaptive planned holding time at entry, not yet a continuous HOLD/SELL
policy.

## Required diagnostics

For every evaluation month report:

- strict date provenance;
- attempts, evaluated and missing-reason counts for all policies;
- GROSS, BASE and STRESS return distributions;
- day-balanced BASE, median, p05, severe-loss rate and worst day;
- selected predicted EV versus realized BASE;
- action-horizon distribution;
- evaluation coverage by chosen horizon;
- for every horizon: probability AUC/Brier/log-loss, hurdle-EV Spearman,
  predicted-positive-EV rate and realized BASE mean;
- primary hindsight-oracle regret and exact-best-horizon rate, clearly labeled
  non-executable;
- primary versus fixed-15 selected-set overlap;
- external and filing-semantic coverage;
- pooled trading-day bootstrap;
- frozen position-ledger BASE/LIGHT/STRESS accounting using the chosen action
  horizon.

## Development promotion rule

Do not access April 2026 or later unless all conditions hold for the primary
adaptive policy:

1. at least 15 evaluated trades in January, February and March;
2. arithmetic BASE mean > 0 in every month;
3. day-balanced BASE mean > 0 in every month;
4. day-balanced BASE mean strictly exceeds the reproduced fixed-15 semantic
   comparator in every month;
5. BASE p05 and STRESS mean are not worse than the fixed-15 comparator in any
   month;
6. pooled trading-day-cluster bootstrap 95% lower bound > 0;
7. chosen-action evaluation coverage is at least 90% in every month;
8. required external/semantic causal coverage remains valid;
9. BASE position-ledger marked return > 0 with complete accounting in every
   month;
10. the reproduced 15-minute comparator matches the frozen v3.0 selected counts
    and monthly BASE means to reporting precision.

A hindsight-oracle profit does not satisfy any promotion condition.

## Failure rule

Do not tune the horizon set, zero-EV rule, model capacity, cost assumptions,
calibration rules or feature subset after seeing v3.2.

If v3.2 fails:

- if horizon diagnostics and chosen-action regret show useful action ranking but
  selected candidates remain negative, separate a causal opportunity/abstention
  head from relative horizon ranking;
- if selected GROSS is positive but BASE is negative, execution-cost
  observability remains the bottleneck;
- if action ranking itself is unstable, improve causal continuation/exhaustion
  state before adding HOLD/SELL freedom.

Only after a cost-positive entry + planned-horizon process replicates should the
research open unrestricted continuous HOLD/SELL policy learning.
