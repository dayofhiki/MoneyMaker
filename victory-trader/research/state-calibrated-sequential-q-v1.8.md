# State selected-tail calibrated sequential Q v1.8

Pre-registered before viewing v1.8 outcomes. January-March adaptive development
only; no fresh month, deployment, execution-cost discount or parameter search.

## Diagnosis
v1.7's 515 executed trades had gross mean +0.343660%, modeled friction
1.174430%, and base mean -0.830770%. Monthly predicted selected Q means were
+0.597/+0.537/+0.667%, versus actual base -0.597/-1.012/-0.939%.
Even the >$10 subgroup was negative. Overall BUY rank correlation (~0.20)
does not establish positive selected-tail value. WAIT correlation was near zero
and negative in February. Adding static external families did not overcome this.

## Intervention
Keep v1.7 feature families, BUY10/WAIT5/SKIP geometry, zero action threshold,
three-stage fit partitions, regressor capacity, winsorization and cost scenarios.
Reserve the chronological final 20% of training trading days as calibration D.
Split remaining training days into the unchanged A/B/C thirds.

Train/calibrate backward:
1. fit Q10_BUY on A, calibrate on D;
2. fit Q5_BUY on B; construct Q5_WAIT targets using calibrated Q10 on B;
   calibrate Q5_BUY and Q5_WAIT on D;
3. fit Q0_BUY on C; construct Q0_WAIT targets using calibrated downstream
   policies on C; calibrate Q0_BUY and Q0_WAIT on D.

For each model, use calibration rows with raw predicted Q >0 and finite labels.
Calculate daily mean optimism error (prediction minus realized value).
Subtract max(0, mean daily optimism +1.645 * standard error across days)
from every prediction. If fewer than five calibration days are represented,
disable that action (adjusted Q=-infinity). Do not relax this after results.
This is a fixed pessimistic bias allowance, NOT a formal state-wise confidence
bound or a guarantee under distribution shift. Reuse of D across stages makes
the experiment exploratory; full fresh-month validation remains necessary.

Batch checkpoint construction and inference must preserve exact timing,
first attempt, tie order and missing-checkpoint semantics. Future returns remain
labels/evaluation only. Missing realized labels are not silently filled with zero;
report evaluated vs attempted coverage.

## Comparators and outputs
Compare earliest eligible BUY10, raw-Q version fitted on identical A/B/C data,
and calibrated sequential policy. This separates calibration effects from the
20% data reservation. Report stage corrections/calibration days, selection bias,
gross vs modeled friction, day-balanced base, tail/stress, paths, common episodes,
and pooled day-cluster bootstrap. Save calibration diagnostics and trade outcomes.

## Decision rule
No promotion unless calibrated policy has >=15 trades each month, positive
base and day-balanced means each month, day-balanced improvement over earliest
each month, p05/stress no worse each month, positive pooled day-bootstrap lower
bound, and unchanged external coverage >=90%/complete.
If calibration yields zero trades, report insufficient validated economic signal,
not a profitable abstention strategy. Do not tune the allowance or timing after
inspection. If positive selection bias persists, add genuinely intraday/execution
information or more development periods under a new protocol rather than rewriting
class labels again.

## Execution record (before complete outcomes)
- Implementation passed full CI: 247 tests and Ruff critical checks, run 35387885404.
- Request 46, run 35388012405 attempts 1 and 2: both completed January, then received a runner shutdown signal during the following fold. No Python model exception or complete outcome artifact was available. The cause of the shutdown is not established; do not treat an incomplete run as an economic model result.
- Removed redundant full-panel references after train/calibration splitting and model fitting, explicitly collected completed-fold buffers, and added per-fold metrics/calibration logging. These execution-only changes do not alter model inputs, splits, predictions, seeds, cost assumptions or decision rules. A fresh request is required to execute this corrected revision.

## Completed outcome — request 47
Run [35389089413](https://github.com/dayofhiki/MoneyMaker/actions/runs/35389089413)
completed successfully on the memory-corrected revision. Artifact
[moneymaker-state-calibrated-sequential-q-v18-47](https://github.com/dayofhiki/MoneyMaker/actions/runs/35389089413/artifacts/10565520659)
has digest `sha256:eb004ac0b980cbc4c047fd50e3d653f3a188e67dd70bd060549fae5d61464dfd`.
The successful retry is evidence that the execution correction was useful,
not proof of the shutdown's underlying cause without resource telemetry.

| Holdout | Raw same-fit trades | Raw base mean | Calibrated trades | Calibrated base mean |
| --- | ---: | ---: | ---: | ---: |
| January | 291 | -0.915769% | 0 | unavailable |
| February | 134 | -0.743722% | 1 | -3.822639% |
| March | 152 | -0.650326% | 0 | unavailable |

Raw same-fit day-balanced means: -0.804399%, -0.952991%, -0.653860%.
Raw selected predicted Q means: +0.657247%, +0.670120%, +0.519866%.
Rank correlations remain positive (+0.158775/+0.205299/+0.224540) while
all raw selected-tail base means remain negative. This is not positive economic EV.

BUY calibration samples covered 8-9 selected days and 21-66 rows depending
on stage/fold. Their daily mean optimism was +0.874253 to +4.060765 percentage points.
The fixed corrections were 2.512765-7.560369 percentage points.
Thus BUY was suppressed by measured optimism/uncertainty, not by failing the
five-day requirement. Downstream BUY suppression yielded zero positive WAIT
calibration rows, so WAIT was disabled as pre-registered. This does not prove
that no tradable edge exists in the market; it says this model/data/cost setup
does not substantiate one.

The calibrated policy had exactly one evaluated day. The original artifact's
one-day bootstrap printed identical bounds at -3.822639%; those are degenerate
resampling output, NOT an informative confidence interval. A reporting-only
guard added after this completed experiment now returns unavailable bounds for
fewer than two evaluated days. This does not change any prediction, trade,
cost, experimental rule or recorded result, and is covered by regression tests.
No additional experimental rerun or parameter relaxation was performed.

External scoreable coverage is identical to v1.7: short-volume 100%,
short-interest latest 98.7145%/98.9399%/98.9049%, and 8-K/short-interest
queries complete. The failure is not attributable simply to absent external data.

**Decision: reject v1.8 as a profitable model candidate.**
Minimum 15 trades/month, positive monthly base/day means, baseline improvements
and usable positive pooled inference were not established. Zero trades are not
profitable performance; the one remaining trade lost money.
The calibration guard is retained as defensive research infrastructure, not
claimed as a demonstrated alpha improvement.

## Next research direction
Do not lower the correction or relabel BUY/WAIT/SKIP again on January-March.
The observed friction deficit, selected-tail optimism and negligible timing
gain require new information and broader development support, not more capacity
alone. A subsequent separately preregistered protocol should:
- audit acquisition and point-in-time availability of intraday liquidity and
  executable-cost information; never infer tradability from future label presence;
- distinguish probability of executable/evaluable entry from conditional net
  return, and report attempted vs evaluated selection (raw: 348/291, 175/134,
  192/152 attempts/evaluated across the three months);
- assess raw loss tails and selected-tail calibration with substantially more
  chronological development days before selecting a new policy;
- lock the new model/cost/data-quality criteria before any genuinely untouched
  month is consumed.

No fresh-month data or live/paper deployment was used or authorized in this run.

Final implementation, including the reporting-only bootstrap guard, passed
[CI 35389770345](https://github.com/dayofhiki/MoneyMaker/actions/runs/35389770345):
250 tests and Ruff critical checks. No production deployment was made.
