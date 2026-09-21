# Expanded-history adaptive-horizon diagnostic v3.1 — pre-registration

## Status

Pre-registered after v3.0 completed and failed, before inspecting any v3.1
multi-horizon diagnostic output. Use only the already-open September 2025 through
March 2026 development research. April 2026 and later remain sealed.

This is a diagnostic experiment, not a candidate strategy and not a promotion
attempt.

## Motivation fixed before results

v2.4-v3.0 repeatedly showed a useful but imperfect distinction between broad
ranking ability and realized selected-tail value. In v3.0 the filing-semantic
model remained sparse and BASE-negative at the fixed 15-minute action, while
the larger research program's final target is an adaptive trader that may hold
different opportunities for different lengths of time.

Before adding HOLD/SELL freedom or training another absolute EV head, determine
whether the fixed 15-minute horizon is materially responsible for the observed
losses.

The diagnostic separates three cases for every evaluated 15-minute anchor:

1. fixed-15 success: realized 15-minute BASE return > 0;
2. horizon miss: realized 15-minute BASE return <= 0, but at least one other
   already-frozen horizon has positive BASE return;
3. candidate miss: realized 15-minute BASE return <= 0 and no already-frozen
   horizon has positive BASE return.

This distinction directly tests whether the next bottleneck is primarily entry
selection or horizon control.

## Frozen input

Reuse the exact evaluated trade rows already produced by successful v3.0 request
79. No model is refit.

Analyze these three frozen policies separately:

- feasible_earliest_15m_cap1;
- split_supply_hurdle_ev_15m_cap1;
- filing_semantic_split_supply_hurdle_ev_15m_cap1.

Evaluation months remain January, February, and March 2026.

## Frozen horizons and outcomes

Use only state-panel labels that were defined before v3.0:

- 1 minute;
- 2 minutes;
- 5 minutes;
- 10 minutes;
- 15 minutes;
- 30 minutes.

For each horizon report GROSS, BASE and STRESS return summaries. Do not add a
new horizon after seeing v3.1.

The per-trade hindsight maximum across horizons is an oracle ceiling only. It
uses future information and may never be described as an executable strategy,
backtest, or validated edge.

## Required diagnostics

For every month and policy report:

- finite-label count by horizon;
- GROSS, BASE and STRESS mean by horizon;
- modeled BASE cost drag = GROSS minus BASE;
- BASE median, p05, positive rate and day-balanced mean;
- fraction of GROSS-positive trades erased to nonpositive by BASE friction;
- fixed-15 positive rate;
- horizon-miss rate;
- candidate-miss rate;
- horizon-miss rate conditional on a nonpositive 15-minute result;
- rate with any positive GROSS horizon;
- rate with any positive BASE horizon;
- rate with some positive GROSS horizon but no positive BASE horizon;
- hindsight-oracle GROSS and BASE means;
- hindsight-oracle BASE lift over fixed 15 minutes;
- day-balanced oracle BASE mean;
- descriptive distribution of the hindsight-best horizon.

## Interpretation rule

No v3.1 statistic promotes a horizon or policy.

If horizon misses are substantial, the next separately pre-registered model may
predict horizon/continue/exit value from causal state. If candidate misses
dominate, the next branch should improve entry/candidate state information before
adding more exit freedom. If GROSS-positive/no-BASE-positive cases dominate,
execution-cost observability remains the bottleneck.

The choice of the next branch must use the complete frozen diagnostic, not one
attractive month or one horizon cell.

## Guardrails

- no April 2026+ access;
- no new feature selection;
- no threshold tuning;
- no model refit;
- no selection based on v3.1 future-return labels;
- no claim that the oracle is executable;
- no change to the existing LIGHT/BASE/STRESS cost assumptions;
- no live, paper, or brokerage action.
