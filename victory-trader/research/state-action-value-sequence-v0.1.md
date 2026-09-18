# Direct Multi-Horizon Action Value with Causal Sequences — Development Result

Date: 2026-09-18  
Scope: January–March 2026 development panels only  
Workflow run: https://github.com/dayofhiki/MoneyMaker/actions/runs/35344337658  
Artifact: `moneymaker-state-action-value-v01-23`

## Question

Does adding strictly causal, ticker-local state history improve the direct base-net action-value model enough to support a stable executable candidate?

The added feature family contains 36 lagged state features:

- six source signals: 1-minute return, HOD distance, regular-VWAP distance, 1-minute volume acceleration, bar range, and bar close location;
- six lags per source: 1, 2, 3, 5, 8, and 13 minutes;
- all shifts are grouped by trading day and ticker;
- future-row perturbation tests confirm that past feature rows do not change.

The evaluation remains leave-one-month-out across the already-seen January–March panels. No fresh validation month was consumed.

## Primary result

At the pre-existing 0.50% predicted base-EV gate:

| Metric | Breadth only | Breadth + sequence |
|---|---:|---:|
| Months with positive base-net mean | 0/3 | 2/3 |
| Months with positive day-balanced mean | 0/3 | 1/3 |
| Median monthly base-net mean | -2.146% | +0.485% |
| Worst monthly base-net mean | -4.320% | -3.040% |
| Aggregate trade-weighted base-net mean | -2.226% | -0.041% |
| Aggregate day-balanced base-net mean | -1.885% | -0.351% |
| Aggregate stress-net mean | -4.283% | -2.097% |

Month detail for the sequence model:

| Month | Trades | Base-net mean | Day-balanced mean | Stress-net mean | 5% tail |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 25 | -3.040% | -2.848% | -5.153% | -19.484% |
| 2026-02 | 87 | +0.485% | +1.211% | -1.542% | -9.273% |
| 2026-03 | 32 | +0.871% | -0.547% | -1.219% | -11.140% |

The causal sequence family therefore improves ranking and selection materially, but it does not yet create robust positive absolute expectancy.

## Robustness diagnostics

Trading-day cluster bootstrap at the 0.50% gate:

| Scope | Day-balanced mean | 95% bootstrap interval |
|---|---:|---:|
| 2026-01 | -2.848% | [-11.561%, +3.887%] |
| 2026-02 | +1.211% | [-1.214%, +3.910%] |
| 2026-03 | -0.547% | [-4.923%, +2.691%] |
| All development months | -0.351% | [-3.085%, +2.018%] |

No interval excludes zero. The apparent improvement is not statistically stable at the trading-day level.

The predicted EV is also not locally well calibrated inside the selected tail:

| Month | Mean predicted base EV | Mean realized base-net | Prediction–realization gap | Within-tail correlation |
|---|---:|---:|---:|---:|
| 2026-01 | +0.654% | -3.040% | -3.694% | -0.049 |
| 2026-02 | +0.712% | +0.485% | -0.227% | +0.008 |
| 2026-03 | +0.719% | +0.871% | +0.152% | -0.179 |

The gate identifies a better region, but fine-grained EV ordering inside that region is effectively absent.

## Concentration and re-entry

At the 0.50% gate, only 36 of 144 selected sequence states overlap the breadth-only selections. Sequence-only states average +0.173% base-net, versus -2.674% for breadth-only states rejected by the sequence model. This is evidence that path history adds useful selection information.

However, repeated entries on the same ticker-day create large correlated exposure. Examples include:

- RUBI on 2026-02-19: 12 trades, -47.504 percentage points of summed base-net return;
- SXTC on 2026-01-09: one trade at -36.154%;
- CREG on 2026-03-17: one trade at -27.603%.

A post-hoc one-entry-per-ticker-day diagnostic at the same 0.50% gate yields:

| Month | Trades | Base-net mean | Day-balanced mean | Stress-net mean |
|---|---:|---:|---:|---:|
| 2026-01 | 18 | -2.737% | -2.521% | -4.911% |
| 2026-02 | 35 | +2.833% | +4.673% | +0.727% |
| 2026-03 | 23 | +0.535% | -0.984% | -1.477% |
| All | 76 | +0.818% | +0.981% | -1.275% |

The all-month day-bootstrap interval remains [-2.502%, +4.517%], so this is a promising risk-control hypothesis, not a validated candidate.

## Decision

Status: **retain the causal sequence feature family; reject promotion to a frozen candidate.**

Reasons:

1. aggregate base-net expectancy is approximately zero;
2. only one of three months is positive after equal weighting by day;
3. all three months are negative under stress friction at the uncapped primary rule;
4. confidence intervals remain wide and cross zero;
5. January remains materially negative;
6. rare single-trade losses dominate some days;
7. the one-entry cap result was inspected post hoc and cannot be treated as confirmation.

## Next pre-registered development experiment

Before consuming a fresh validation month:

1. keep the 36 causal sequence features unchanged;
2. keep the 0.50% base-EV gate unchanged;
3. enforce a maximum of one entry per ticker-day as a portfolio-risk constraint;
4. add a train-only severe-downside probability model using the already pre-specified -5% severe-adverse threshold and 20% probability gate;
5. evaluate uncapped/no-risk, capped/no-risk, and capped/risk-gated variants together;
6. report trade-weighted, ticker-day-balanced, and day-balanced expectancy plus cluster bootstrap intervals and stress friction;
7. do not tune thresholds after seeing January–March results.

Promotion requires at minimum:

- positive base-net mean in at least two of three development months;
- positive day-balanced mean in at least two of three months;
- positive aggregate base-net and day-balanced means;
- reduced severe-loss frequency without collapsing trade count;
- no result dependent on one day or one ticker-day;
- materially improved stress-friction performance.

Only after the rule is frozen should the next untouched validation window be consumed.
