# R306 findings — ranking improves, probability quality fails

Official run37211254516; source1566487228e26419037683ec48c7768f4717b873;
artifact11307150821. Execution and remote CI succeeded.39 selected tests passed.
All97 original episodes and4526 states have raw data and finite labels. Exactly
11 missing original May ticker-days were fetched once, with11 network requests,
zero retries and no June access.86 frozen reference entries are unchanged.

## Bottleneck evidence

1079 states have positive original MFE+MAE utility but no reachable net profit.
898 states have whole-path net+5 upside that becomes unavailable under absorbing
risk. State counts are correlated observations, not1079/898 independent trades.
The old utility has only0.417996 rank correlation with the risk/cost ceiling.
Entry fill gaps median0s,p90 10s,p99 62.75s,max238s remain in labels; next-print
references are assumptions, not tradable quote guarantees.

| First cash net+5 model | AUROC | AP | Brier |
|---|---:|---:|---:|
|A old utility,75 causal features|0.397903|0.077484|not a probability|
|B risk/cost target,75 features|0.551921|0.106336|0.098203|
|C risk/cost target,99 clock/cost features|0.597127|0.127366|0.103026|
|D risk/cost target,76 cost-only features|0.527301|0.100367|0.100021|
|Outer-training state prevalence benchmark|—|0.090219|0.081417|

First snapshot winners9/97. May5 has no positive class and is unevaluable for
within-day rank; C beats B/D on all3 other days. C-B ticker-day AUROC gain95%
CI[-0.046577,+0.135866]; C-D[-0.000740,+0.139490]. Day-bootstrap gains exclude
zero but only four repeatedly used days and one zero-class day constrain them.
Primary preregistered gate FAILED because C Brier worsened B and cost-only D.
A modest ranking gain is evidence for the new representation, not a validated
probability estimate, optimal entry, profitable policy or completed model.

Clock20s snapshot C AUROC0.650516 vs B0.555618/D0.531110,92/97 available;
clock60s C0.586023 vs B0.528699/D0.550861,88/97; clock120s C0.578327 vs
B0.607886/D0.626650,82/97. Comparisons are matched within each clock snapshot,
not across changing populations. They do not select20s as an entry delay or
prove later waiting profitable. No forced wait, threshold tuning or promotion.

The99-feature package includes completed-clock return/efficiency, activity,
volume/transaction acceleration, signed-volume PROXIES and prior-high reclaim
alongside known cost. D controls for cost alone; it does not identify a specific
individual momentum feature as causal. Missing120s window baselines remain NaN.
Saved learned upstream/model scores are excluded from new feature allowlists.

Next R307 isolates training-only monotone probability calibration using nested
inner-day OOF predictions and first cash snapshots. Keep all model/features/risk
fixed. Constant fallbacks or worsened rank cannot be called learned momentum.
June remains sealed and no trading policy is changed. See results/request306.json.
