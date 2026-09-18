# State Rank-Turn Stop v0.6 — pre-registration

## Status

Pre-registered before implementation and before viewing any v0.6 result.

This experiment uses only the already-seen January-March 2026 development
panels. It must not load or inspect a fresh validation month.

## Motivation

Three development results now separate two kinds of timing signal:

- State Entry Ranker v0.3 learned meaningful out-of-month within-episode
  relative ordering, with positive Spearman in every month and horizon, but its
  fixed top-25% absolute score gate was too sparse.
- State One-Step Stop v0.5 found almost no predictable next-minute return
  derivative and did not improve the baseline.

The next question is whether the stronger episode-rank signal can be converted
into an executable policy without another tuned absolute score threshold.

## Fixed information set and models

Reuse the State Entry Ranker v0.3 information set and models unchanged:

- calibrated direct base-net EV regressors for 5-, 10-, and 15-minute actions;
- episode-normalized rank regressors for the same three action horizons;
- point-in-time state features;
- contemporaneous leave-one-out runner breadth;
- the existing 36 causal ticker-local lag features;
- current and previous absolute price transforms.

The direct buy gate remains predicted base-net EV >= 0.50%.

No new data source, news feature, risk classifier, execution assumption, or
fresh validation month is allowed.

## Current rank score

At each state:

1. retain only action horizons whose predicted direct base EV is >= 0.50%;
2. among those horizons, choose the horizon with the highest predicted
   episode-rank score;
3. call that maximum the current rank score.

This is exactly the rank-action construction already used in v0.3. No absolute
rank-score gate is applied.

## Causal policies

### earliest_ev_cap1

Exact v0.3 baseline: enter the first chronological state in a ticker-day whose
best predicted direct action EV is >= 0.50%, choosing the highest direct-EV
horizon. Permit at most one attempted entry per ticker-day.

### rank_turn_cap1

For each ticker-day, process states chronologically.

1. The first EV-qualified rank state arms the tracker but does not enter.
2. A comparison is valid only when the immediately preceding tracked state is
   exactly one minute earlier and is also EV-qualified.
3. If the current rank score is strictly greater than the previous tracked
   score, continue waiting.
4. At the first valid state whose current rank score is <= the previous tracked
   score, enter at the current state using the current rank-selected horizon.
5. If EV qualification breaks or the timestamp gap is not exactly one minute,
   reset the tracker. A newly qualified state becomes a new first tracked state.
6. Permit at most one attempted entry per ticker-day.

There is no rank quantile, learned cutoff, score-difference threshold, maximum
wait hyperparameter, or retrospective peak selection. The rule is a
threshold-free first-turn stopping rule.

If the first accepted turn signal is unfillable or lacks the selected realized
label, the attempted entry is consumed and the policy must not fall through.

## Evaluation

Leave one month out, train on the other two, and repeat for January, February,
and March.

Report:

- trades, days, and ticker-day episodes;
- base, gross, and stress mean return;
- day-balanced base mean;
- p05, severe-loss rate, and worst day;
- entry minute and action-horizon mix;
- unchanged holdout episode-rank diagnostics by horizon;
- fraction of baseline EV-qualified episodes that obtain a causal rank-turn
  entry;
- common-episode entry delay and realized-return delta versus earliest_ev_cap1;
- pooled trading-day-cluster bootstrap interval for rank_turn_cap1.

## Success rule

Do not promote or consume a fresh month unless rank_turn_cap1:

1. has at least 15 trades in every month;
2. has positive base-net mean in every month;
3. has positive day-balanced base-net mean in every month;
4. improves day-balanced base-net mean over earliest_ev_cap1 in every month;
5. does not worsen p05 or stress-net mean versus earliest_ev_cap1 in any month;
6. retains positive episode-rank Spearman in every holdout month for all three
   action horizons; and
7. has a pooled trading-day-cluster bootstrap 95% lower bound above zero.

If this exact branch fails, do not add score thresholds or tune a turn-size
criterion on January-March.
