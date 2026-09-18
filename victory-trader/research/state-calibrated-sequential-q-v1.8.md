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
