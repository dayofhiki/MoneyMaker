# R305 findings — entry economics and early observability both fail

Official run37210193519 completed successfully on source7fc7e54f41b6a7196f00f616e04f99a93842d92c; artifact11306800230. Official economics, labels, rankings and integrity checks match the local result;13 Brier fields differ only by floating-point roundoff, maximum1.12e-16.
Dependencies R304 run36883479409/artifact11172631677 and R298
run36801235949/artifact11136176759. Exact86 R304 B returns/submissions/reasons
reproduced with zero error. All integrity checks pass; no policy promotion.
Selected74 tests and critical research lint pass. No model/feature/threshold/seed
changes followed these results. Censored-report serialization was hardened and
unit-tested; no position in this development run is censored.

## Frozen-entry economics

| Category | Positions | R304 mean BASE net % | Hindsight reachable maximum mean % |
|---|---:|---:|---:|
| No feasible net profit |53|-2.225413|-1.715032|
| Feasible profit missed |29|-1.167200|7.386667|
| Positive profit captured |4|2.881716|9.986720|

All86 resolve.11 have no eligible HOLD state, including9 cost-only stops; they
remain in economics. R304 actual mean -1.631033%; hindsight bound +1.898413%.
The latter requires knowledge of the best future sale and is NOT a backtest,
a teacher, or an achievable expected return. Under STRESS even that bound
averages -0.401621%. The bounds use existing completed-second submissions,
absorbing stop/cap and their first subsequent regular-print open, not raw highs.

| Target | Reachable gross opportunities | Reachable BASE-net opportunities | Whole-path BASE-net opportunities, ignoring stop |
|---|---:|---:|---:|
|+5%|14|12|26|
|+10%|5|4|10|
|+20%|2|1|3|

Ignoring the stop is a descriptive counterfactual, not a reason to relax it.
The53/86 lack positive net payoff under current entries, submission schedule,
stop, costs and print-fill assumption; they are not necessarily intrinsically
bad stocks.29/86 still identify material exit mistakes. Neither entry-only nor
exit-only diagnosis explains the whole failure.

## Causal observability

PRIMARY uses separate first-HOLD-snapshot logistic fits trained on the other
three days, fixed28 causal features and no class balancing.75 eligible snapshots;
11 mandatory-risk trajectories excluded from learning only, never economics.

| Future reachable net target | Positive episodes | Primary AUROC | State-tree AUROC at first decision | State-tree AUROC on R304 visited states |
|---|---:|---:|---:|---:|
|Cost coverage (>0)|33|0.556709|0.566606|0.575477|
|>=+5%|12|0.306497|0.527162|0.529802|
|>=+10%|4|0.223254|0.555592|0.587894|
|>=+20%|1|0.121053|0.121053|0.121053|

Primary +5% ticker-day bootstrap AUROC95% interval[0.164728,0.476135].
Day AUROC +5% May5–8:0.071429,0.375000,0.477273,0.613636. Primary probability
Brier0.163678 versus training-prevalence baseline0.143120 (lower is better).
AP0.180121 versus baseline0.130042 does not override failed rank/calibration:
pooled AP can benefit from fold/day prevalence or an isolated top-ranked winner.
Do not invert the failed scores or tune thresholds after seeing the results.
+10/+20 support is too small to demonstrate transfer; the only net+20 winner's
held-day fold has zero training positives and therefore a constant fallback.

All10618 counterfactual reachable states show secondary +5% AUROC0.776029,
but conditioning on observed net still below5% lowers it to0.686966. Visited
states remain0.529802. Later unvisited states are NOT evidence that the current
policy can safely wait to obtain their information. State net+5% probability
Brier remains worse than baseline, including the not-yet-reached subset.
The one inherited risk-reachable row at the exact cap is a terminal submission,
not a voluntary HOLD opportunity, hence10618 versus R304's10619 marked rows.
Regression ranking of maximum EXIT advantage is negative at first states
(-0.084013 day/episode-weighted) and R304 visited states(-0.095985).

## Next design priority — R306, not dispatched here

Audit the PRE-ENTRY observation loop on the already-opened May5–8 candidate
set, including11 R296 abstentions and11 immediate-risk positions, rather than
another HOLD teacher/pi iteration. Recover the saved causal entry states and
raw seconds, retaining the same shortlist and frozen86-entry reference. Ask
whether cash WAIT/recheck states BEFORE paying costs supply distinguishable
cost-cover and >=5% opportunities before the existing entry window/common cap.
Evaluate preregistered first/fixed recheck snapshots with day-held-out fitting;
report missing/gap states and liquidity/cost-only-stop support explicitly.

Any later-time entry ceiling remains a diagnostic, not selecting the best future
entry or forcing a purchase. Existing stop/cost/print rules stay frozen. A new
entry action is justified only after causal separation and payoff are shown;
otherwise broaden representation/opportunity supply on development data.
Do not force a minimum HOLD, relax risk, remove losing positions or open June.
The target remains recurrent SCAN/WATCH/WAIT/ENTER/HOLD/EXIT, with positive
net economic value, not a fixed-horizon or peak-picking strategy.

Four reused days, conditional upstream entry OOF and next-print assumptions
limit every result. Event means are not account returns. See results/request305.json.
