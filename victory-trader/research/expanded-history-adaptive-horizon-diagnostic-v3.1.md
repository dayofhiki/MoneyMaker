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


## Result — request 82

Workflow run 35594971454 completed successfully. Artifact
`moneymaker-expanded-history-v31-82` has digest
`sha256:426d0c5bec8161506b37cd0ed19889eb7d3406490f69f4c72db2824e591e133e`.

The diagnostic reattached only the already-frozen state-panel outcomes at the
exact v3.0 evaluated trade keys. No model was refit and April 2026+ remained
sealed.

### Primary v3.0 semantic tail

| Month | Trades | 15m BASE | Any BASE-positive horizon | Horizon miss | Candidate miss | Oracle BASE |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | 48 | -0.526% | 75.0% | 37.5% | 25.0% | +2.881% |
| 2026-02 | 6 | -2.472% | 33.3% | 0.0% | 66.7% | -1.052% |
| 2026-03 | 4 | -6.058% | 25.0% | 25.0% | 75.0% | -2.513% |

In January, 60% of the trades that lost at the fixed 15-minute horizon had at
least one other frozen horizon with positive BASE return. The per-trade
hindsight horizon ceiling improved mean BASE by +3.408 percentage points.

That pattern did not transport to February or March. The hindsight oracle itself
remained negative in both months, so horizon choice alone could not rescue those
selected tails. Their dominant failure was candidate/state selection.

Across all 58 primary trades, the diagnostic split was nearly even: 20 fixed-15
winners, 19 horizon misses, and 19 candidate misses. This pooled split is
descriptive only because January contributes most observations.

### Broad execution-feasible anchors

The broad first-anchor universe showed a stable but difficult opportunity:
any frozen horizon was BASE-positive for 47.3% / 47.6% / 44.9% of anchors in
January / February / March, while the hindsight best-horizon BASE mean was
+1.138% / +0.831% / +1.091%. Fixed 15-minute BASE means remained about
-1.4% to -1.8%.

This is not an executable edge. It shows that the data contain state-dependent
holding-horizon opportunity, while roughly half the anchors have no profitable
frozen horizon after BASE costs.

## Decision

v3.1 confirms that fixed 15 minutes is a real bottleneck for some opportunities,
especially January, but it is not the sole or dominant explanation for the
cross-month failure.

The next branch should therefore combine both decisions rather than merely swap
15 minutes for another fixed horizon:

1. estimate action-conditioned upside/downside and BASE expected value for the
   frozen 1/2/5/10/15/30-minute actions from the same causal state;
2. abstain when no action has positive predicted BASE EV;
3. otherwise choose the horizon with the highest predicted BASE EV;
4. evaluate the actually chosen horizon, not the hindsight oracle;
5. report max-action optimism and unevaluable-action coverage explicitly.

This is the next intermediate step toward a continuous HOLD/SELL trader. It
tests whether causal state can choose both whether to trade and how long to hold
before adding unrestricted exit freedom.
