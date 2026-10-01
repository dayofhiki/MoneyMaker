# R297 findings — HOLD/EXIT gate failed; repair execution semantics first

Source: completed research run 36798494177, commit
4cf66187ce6da9033a2b6a54116c8b20860391d5, artifact 11134922041.
May5-8 only; no fresh June HOLD data opened. 97 candidate episodes, 86 entries,
37,627 position states. Entry gates remained frozen from R296.

## Observed results

All returns below are BASE net means **among resolved positions only**.
These subsets differ and must not be compared as unconditional strategy returns.

| Policy | Resolved / 86 | Resolved mean | Mean hold, resolved |
| --- | ---: | ---: | ---: |
| Immediate | 37 | -1.1801% | 1.76s |
| Fixed 60s | 36 | -1.3697% | 51.81s |
| Fixed 300s | 35 | +0.4337% | 250.06s |
| Fixed 600s | 37 | +0.4808% | 347.22s |
| Old 10m / 2% trailing | 33 | -1.7689% | 135.58s |
| Dynamic 60s | 38 | -1.0247% | 27.79s |
| Dynamic 300s, primary | 38 | -1.1075% | 79.55s |

Primary dynamic policy was negative on all four days, and also negative with
LIGHT costs (-0.3422%). Its 95% resolution gate failed (44.19%). Candidate
cash-adjusted means were withheld because unresolved positions are not cash.

On the **same 27 resolved episodes**, dynamic 300s improved versus old trailing
by only +0.0061 percentage points. Day bootstrap CI95: [-0.2003, +0.3711]pp;
ticker-day CI95: [-0.4784, +0.4617]pp. No convincing improvement.
Matched versus immediate: dynamic 300s -0.0465pp on 27 episodes; dynamic 60s
-0.0850pp on 31 episodes. These are selected matched subsets, not full-population
estimates.

The fixed 300s positive mean depends on May8 STFS (+75.4984%, held 3373s).
Removing that one observation changes +0.4337% to -1.7740%.
For fixed 600s, removing it changes +0.4808% to -1.6030%.
A nominal five-minute policy held this trade for over 56 minutes: the timeout
was evaluated only when another active second arrived. This is a clock/order
simulation defect, not evidence for five-minute profitability.

## Additional diagnostics from saved states

- First-state execution within 3s: 37/86 = 43.02%.
- All-state executable rate: 73.80%.
- Following-open lag median 1s, p75 4s, p90 14s, p95 29s, p99 105.74s.
- Terminal complete paths under the exact endpoint/3s requirement: only 4/86.
  Consequently +5/+10/+20 capture and left-on-table reports cover extremely
  few terminal-complete paths. They cannot describe the full opportunity set.
- 60s HOLD target coverage 59.93%; OOF prediction/target Spearman -0.0204.
- 300s HOLD target coverage 56.74%; OOF Spearman -0.0718.
  Neither horizon demonstrated predictive ranking on the available labels.
  State rows are dependent; these are diagnostics, not independent evidence.
- At unchanged price, BASE round-trip friction alone breaches the -2.5% net
  hard stop for 9/86 entries. Median unchanged-price cost is 1.1428%.
  Eleven entries trigger the stop on their first completed state. Cost-only
  breaches and observed first-state breaches are separate quantities.

## Interpretation

R297 did not learn a demonstrated profitable HOLD/EXIT policy. It also exposed
an inadequate execution simulator for sparse active-second data:

1. An order whose next print arrives later than 3s was immediately classified
   unresolved rather than tracked as pending. Excluding these positions changes
   the sample differently for every policy.
2. A clock deadline was checked only at market-data events, allowing timeouts
   to move through long silent gaps.
3. Exact terminal execution was unnecessarily entangled with the availability
   of an observed-window upside diagnostic; only four paths contributed.
4. Net-loss stopping interacts with substantial minimum-cent friction in cheap
   stocks. This is real under the scenario, but is not solely price deterioration.

These are separate from model quality. Fixing them may change returns in either
direction. It does not establish that the negative HOLD signal will recover.
Upstream OOF training is still not nested inside HOLD folds; even a repaired
May result would be exploratory. June and July-August remain sealed.

Next: R298 execution-integrity and label-coverage audit, with unchanged entry,
features, horizons, thresholds, costs and stop logic. Do not select a new HOLD
architecture from the broken R297 headline means.
