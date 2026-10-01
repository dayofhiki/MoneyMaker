# R300 findings — clock/context ranking does not improve realized exits

Official run: https://github.com/dayofhiki/MoneyMaker/actions/runs/36830074540
Source: 04a372b93ef206ff747c1b7ddcd01c8bf545a052.
Artifact: 11146592468, moneymaker-research-request-300.
Execution succeeded, 35 tests passed, economic gate FAILED. Official JSON equals
the complete local verification JSON. No post-result parameter changes.

## Same-entry economics

All arms resolve the same 86 positions, with exact R299 baseline reproduction.
Values are per-position means, not compounded account returns.

| Arm | BASE mean % | Gross mean % | Positive positions | Median hold s |
|---|---:|---:|---:|---:|
| A: R299 D | -1.346767 | +0.197371 | 9/86 | 13.0 |
| B: add elapsed-time features | -1.476840 | +0.066094 | 6/86 | 9.5 |
| C: add close-range context, primary | -1.473654 | +0.069138 | 5/86 | 9.0 |

C-A -0.126887pp; C-B +0.003186pp (negligible and uncertain).
C-A day bootstrap CI [-0.220570,-0.046761], ticker-day
[-0.278230,+0.005478]. Four reused days do not establish generalization.
C daily means all negative: -2.035789, -1.130933, -1.915426, -1.120928%.
LIGHT -0.638296%, STRESS -3.708757%, largest-winner-excluded -1.548508%.
CVaR5 worsens A -5.088428% -> C -5.251287%. Common cost drag remains about
1.54pp; deterioration is primarily lower gross capture, not increased friction.

## Ranking versus decisions

Archived weighted rank improves A -0.028716 -> C +0.056993; first-decision
rank A +0.090218 -> C +0.167739. However each feature arm fits its own pi1,
so its continuation labels also change. These ranks are not a common-label
measurement of feature improvement. Diagnostic comparison on identical saved
A-pi1 labels yields A -0.028716, B -0.042359, C +0.015931. On common frozen
pi0 labels ranks remain negative: A -0.066819, B -0.027247, C -0.016825.
C's May8 weighted rank remains -0.202979. Ranking alone cannot validate the
zero-threshold HOLD/EXIT action, much less profitability after entry costs.

Only 19/86 C realized returns differ from A: six improve, thirteen deteriorate.
Illustrative paired changes (BASE net %, next-print reference execution):

| Ticker/day | A return | C return | A hold s | C hold s |
|---|---:|---:|---:|---:|
| JDZG May7 | +1.648785 | -1.214611 | 144 | 1 |
| DGXX May6 | +1.299208 | -1.169740 | 19 | 1 |
| BGDE May6 | +1.645108 | -0.429282 | 7 | 1 |
| ANY May7 | -1.001297 | -4.011499 | 28 | 23 |

These examples describe realized outcomes, not rules to hold these tickers next
time. Not every deterioration is early exit; ANY and PYXS show other changes.
56 C model exits occur at the first decision versus 49 for A. C submits 81/86
exits within 60s of entry and 76/86 within 20s. Of 72 model-exit states, 69 have
missing new 60s context features because the preregistered feature prefix begins
at entry and requires 60 elapsed seconds. Missingness itself can affect a tree;
this does not mean these inputs have no effect or prove it caused early exit.

33/72 model-exit states have positive C-pi1 continuation labels, mean +0.134904pp
across those 72 exit states, while mean predicted advantage is -0.567093pp.
This flags action calibration/target alignment for investigation; pi1 future
outcomes are diagnostic counterfactuals, not optimal HOLD targets or assured profit.

## Opportunity and execution limits

Archived risk-compatible opportunity diagnostic counts gross +5/+10/+20 paths
as 14/5/2, all with zero corresponding realized NET threshold captures. It
includes the mandatory-stop submission's next-print fill; therefore these counts
differ from R298's strictly-before-stop diagnostic. Do not compare them as if
their definitions were identical. Whole-path opportunities remain 28/12/4.
50 C exits wait more than 3s for a print. Fully reconciled print-based results
still do not prove quote-level fill feasibility. Nine cost-only stop breaches
remain included. No production promotion or opening June.

## Next research direction, not launched

Test causal pre-entry chart context before adding model complexity or imposing
a minimum hold. A trader has observed the chart before buying; current added
features deliberately discard that history. Existing R298 raw data has completed
pre-entry bars for all 86 positions; all have prefix spans >=60s, 66 have >=20
pre-entry bars (median 37.5). Sparse spans do not guarantee dense recent coverage.

A subsequent preregistration should compare the exact C baseline with the same
eight features computed from completed regular-session bars before AND after
entry, using an explicit history-availability contract and the same nested
learning/risk/execution. Separate common-label ranking, sign calibration and
actually visited first decisions. No future backfill, fixed 60s forced holding,
entry/stop retuning, or post-hoc winner selection. June remains sealed. This
hypothesis is supported by feature availability, not yet by profit evidence.
