# State Entry Ranker v0.3 — pre-registration

## Status

Pre-registered before implementation and before viewing any v0.3 result.

This experiment uses only January-March 2026 development panels. It must not load
or inspect a fresh validation month.

## Question

The sequence action-value model appears to find a better region but cannot rank
states precisely inside that region. Test whether a model trained on relative
entry quality within each ticker-day can improve causal entry timing.

## Fixed information set

- point-in-time state features;
- contemporaneous leave-one-out runner breadth;
- the existing 36 causal ticker-local lag features;
- current and previous absolute price transforms.

No news, future bars, fresh month, severe-loss classifier, dynamic stop branch, or
new execution assumption is allowed.

## Models

For each 5-, 10-, and 15-minute action:

1. retain the existing calibrated direct base-net EV regressor;
2. fit a separate histogram gradient-boosting regressor to the action's
   within-ticker-day percentile rank of realized base-net return;
3. construct rank labels inside training ticker-days only;
4. sample training states at the existing three-minute cadence;
5. use the chronological final 20% of training days only to set the rank gate.

The rank gate is fixed at the 75th percentile of calibration-period predicted
rank scores among actions whose predicted base EV is at least 0.50%.

## Causal policies

- `earliest_ev_cap1`: enter the first state in a ticker-day whose best predicted
  action EV is at least 0.50%; choose the highest-EV horizon.
- `rank_top25_cap1`: among actions with predicted EV at least 0.50%, choose the
  highest predicted-rank horizon at the current state and enter the first state
  whose rank score also clears the train-only top-25% gate.

Both policies permit at most one attempted entry per ticker-day. They act on the
first chronological signal. The rank policy must not retrospectively select the
highest score observed later in the day.

## Evaluation

Leave one month out, train on the other two, and repeat for January, February,
and March.

Report:

- trades, days, and ticker-day episodes;
- base, gross, and stress mean return;
- day-balanced base mean;
- p05, severe-loss rate, and worst day;
- entry minute and horizon mix;
- within-episode rank correlation by horizon;
- common-episode timing and return deltas;
- day-cluster bootstrap interval computed only after the fixed run.

## Success rule

Do not promote or consume a fresh month unless `rank_top25_cap1`:

1. has at least 15 trades in every month;
2. has positive base and day-balanced mean in every month;
3. improves day-balanced mean over `earliest_ev_cap1` in every month;
4. does not worsen p05 or stress mean in any month; and
5. has a pooled day-cluster bootstrap lower bound above zero.

If it fails, do not tune the 75% gate on January-March. Retire this exact branch
or change the information set.


## Result

Workflow run: https://github.com/dayofhiki/MoneyMaker/actions/runs/35349699710  
Artifact: `moneymaker-state-entry-ranker-v03-26`

The exact pre-registered branch failed its promotion rule.

The relative rank model itself learned a reproducible out-of-month ordering signal:

- 5-minute global Spearman: 0.068 to 0.083 across the three holdouts;
- 10-minute global Spearman: 0.100 to 0.122;
- 15-minute global Spearman: 0.122 to 0.150;
- median within-episode Spearman was positive for every month and horizon;
- 77% to 87% of evaluable ticker-day episodes had positive within-episode
  Spearman, depending on month and horizon.

However, the fixed train-only top-25% gate was too sparse for the pre-registered
policy. `rank_top25_cap1` produced only 2, 11, and 6 trades in January,
February, and March respectively, versus the required minimum of 15 trades in
every month. The branch therefore fails before return-based promotion criteria
are considered.

The sparse selected samples had positive base-net means in all three months
(+2.008%, +1.985%, +2.027%), but those values are not promotable evidence because
the sample-size rule was fixed in advance and was not met. Common-episode return
deltas were also mixed (-1.572%, -1.424%, +2.710%).

## Decision

- Retire the exact `rank_top25_cap1` branch.
- Do not tune the 75th-percentile gate on January-March.
- Retain the empirical finding that causal state history contains relative
  entry-timing information.
- Do not consume a fresh validation month.
- Move to a new pre-registered optimal-stopping target that asks whether waiting
  from the current state has positive expected value, rather than trying another
  absolute rank-score cutoff.
