# R299 findings — risk alignment helps losses, not profitable HOLD decisions

Official run: https://github.com/dayofhiki/MoneyMaker/actions/runs/36810713498
Source: a37a7bd27ca0e1d5ba7c045e7c853eb77039ef2e
Artifact: 11139453542, moneymaker-research-request-299.
Both execution jobs succeeded; economic research gate FAILED. Official JSON
matches the complete local verification JSON exactly. No outcome-driven tuning.

## Controlled comparison

All arms reconcile and resolve the same 86 entries / 97 candidates. R298 A
reproduces exactly, including submissions and reasons; return error 0.
37,627 states, 10,619 reachable HOLD states, 27,008 forced/post-stop states.
Four reused May5-8 days only; June HOLD and July-August remain sealed.

| Arm | BASE net mean % | Increment pp | Weighted ranked signal |
|---|---:|---:|---:|
| A: R298 replay | -1.476812 | reference | not recomputed |
| B: reachable training rows | -1.500209 | B-A -0.023398 | +0.138633 |
| C: risk-consistent horizon labels | -1.379848 | C-B +0.120361 | -0.087119 |
| D: nested next-event policy improvement, primary | -1.346767 | D-C +0.033081 | -0.028716 |
| C60: preregistered sensitivity only | -1.508961 | not a selection arm | not computed |

D improves on A by +0.130045pp, but exploratory paired bootstrap 95% intervals
include zero: episode [-0.190830,+0.454769], day [-0.145970,+0.398938]. These
additional intervals use 20,000 resamples, seed 20263000, and entry-weighted means.
The archived D-vs-old gain is +0.341227pp: day CI [+0.085550,+0.702020], ticker-day
CI [-0.079629,+0.765837]. Four day clusters do not establish reliable superiority.

## Primary D economics

- BASE -1.346767%, LIGHT -0.510667%, STRESS -3.584002% per resolved position.
- Gross mean +0.197371%; average gross-to-BASE drag 1.544138pp. Lower friction
  also remains negative, so cost is substantial but not the sole explanation.
- All four days negative: -1.882054, -0.998308, -1.650883, -1.099608%.
- Median -1.187451%; positive 9/86; largest winner WTF +4.888908%; excluding it
  leaves -1.420128%. No positive concentration-adjusted economics.
- 18 hard stops / 68 model exits; average hold 95.60s but median only 13s.
- CVaR5 improves from A -6.380146% to -5.088428%; this is loss mitigation.
- Net +5/+10/+20 realization rates all zero under the existing whole-path
  opportunity denominators. These denominators include post-stop opportunities;
  they are descriptive, not fully risk-reachable upside sets.
- Historical next-print resolution is 100%, not proof of executable quotes:
  51/86 D exits wait >3s, including two >300s. 31-300s bucket mean -2.672877%.
  Delay-conditioned results use future realized delay and cannot be entry filters.
- Nine cost-only stop breaches remain counted; their mean is -3.415406%.
  Remaining 77 entries still average -1.104978%; excluding them is diagnostic only.
- Eleven entries have no reachable first HOLD state; 75 contribute HOLD states.

## Signal and causal interpretation

D unweighted Spearman -0.048232, day/episode-weighted rank -0.028716. Daily
weighted ranks: May5 +0.209055, May6 -0.075121, May7 -0.003015, May8 -0.203235.
10,619 rows are correlated observations from 75 positions, not 10,619 independent
trades. Correct row/label boundaries improve tail loss but do not establish
learnable profitable continuation. B's positive weighted rank alone is not a
reason to select B after seeing the primary failure.

R297's return_sum_5/20, positive_fraction_20 and volume_ratio_5_20 are rolling
OBSERVATION counts, not seconds. On D's reachable states, active_gap_s median
2s, p90 16s, p99 100s, maximum 3,086s. Among 9,696 reachable rows with a full
20-observation history, first-to-last span median 40s, p90 334s, p99 1,417.55s,
maximum 3,174s; 37.61% span >60s. This is causal evidence of inconsistent feature
time scales, not evidence that correcting them will generate profit.

## Decision

Do not promote D, tune stops/costs, select B/C post hoc, or open June. R300 will
test explicit elapsed-time features and rolling price context, keeping the R299
targets, nested evaluation, entries, risk and execution fixed. Failure would
justify a separately designed target-stability / upstream-economics audit rather
than more same-four-day parameter searching.
