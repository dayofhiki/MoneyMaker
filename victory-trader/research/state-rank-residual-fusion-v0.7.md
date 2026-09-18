# State Rank-Residual Fusion v0.7 — pre-registration

## Status

Pre-registered before implementation and before viewing any v0.7 result.

This experiment uses only the already-seen January-March 2026 development
panels. It must not load or inspect a fresh validation month.

## Motivation

State Entry Ranker v0.3 and State Rank-Turn Stop v0.6 both showed that the
episode-normalized rank model retains positive out-of-month ordering signal in
every development month and every 5/10/15-minute horizon.

Two hand-written conversions of that signal have failed:

- a fixed top-25% absolute rank gate was too sparse;
- a threshold-free first-turn stopping rule was also too sparse and did not
  consistently improve common-episode realized return.

The next question is therefore not where to place another stopping threshold.
It is whether the rank prediction contains information about the residual error
of the existing direct EV model and can improve the EV estimate itself.

## Fixed information set

Reuse the v0.3 information set unchanged:

- calibrated direct base-net EV regressors for 5-, 10-, and 15-minute actions;
- episode-normalized rank regressors for the same horizons;
- point-in-time state features;
- contemporaneous leave-one-out runner breadth;
- the existing causal ticker-local lag features;
- current and previous absolute price transforms.

No new data source, news feature, risk classifier, execution assumption, or
fresh validation month is allowed.

The execution gate remains +0.50% predicted base-net EV. There is no absolute
rank threshold and no rank-turn rule.

## Cross-fitted rank-residual correction

For each outer leave-one-month-out fold, let A and B be the two development
months used for training and H the held-out month.

For each action horizon separately:

1. Fit the existing direct-EV model and episode-rank model on month A only and
   score month B.
2. Fit the same two models on month B only and score month A.
3. Concatenate those cross-fitted predictions. No row may be scored by a base
   model trained on its own month.
4. For rows with a finite realized base-net return, direct-EV prediction, and
   episode-rank prediction, define residual =
   realized base-net return - predicted direct EV.
5. Fit a one-dimensional monotone isotonic regression from predicted episode
   rank to residual, with out-of-range predictions clipped to the fitted
   boundary. No manually chosen bins, rank cutoffs, or smoothing parameter are
   allowed.

Then fit the existing direct-EV and rank models on A+B exactly as before and
score H.

For each horizon in H:

fused EV = direct predicted base-net EV + cross-fitted isotonic residual
correction(rank prediction).

The correction is allowed to become flat if rank contains no useful residual
information.

## Policies

### earliest_ev_cap1

Unchanged v0.3 baseline. At each ticker-day, enter the first chronological state
whose highest direct predicted base-net EV across 5/10/15 minutes is >= +0.50%.
Use the highest-direct-EV horizon. Permit at most one attempted entry per
ticker-day.

### rank_fused_ev_cap1

At each state, compute fused EV for all three horizons. Enter the first
chronological state in a ticker-day whose highest fused EV is >= +0.50%, using
the highest-fused-EV horizon. Permit at most one attempted entry per ticker-day.

The first signal is the attempted entry. If it is unfillable or lacks the
selected realized label, consume the attempt and do not fall through.

## Evaluation

Leave one month out, train on the other two, and repeat for January, February,
and March.

Report:

- trades, days, and ticker-day episodes;
- gross, base-net, and stress-net mean return;
- day-balanced base-net mean;
- median, p05, severe-loss rate, and worst day;
- action-horizon mix and median entry minute;
- unchanged holdout episode-rank diagnostics by horizon;
- cross-fitted training residual-versus-rank diagnostics by horizon;
- common-episode entry delay and realized-return delta versus earliest_ev_cap1;
- pooled trading-day-cluster bootstrap interval for rank_fused_ev_cap1.

## Success rule

Do not promote or consume a fresh month unless rank_fused_ev_cap1:

1. has at least 15 trades in every month;
2. has positive base-net mean in every month;
3. has positive day-balanced base-net mean in every month;
4. improves day-balanced base-net mean over earliest_ev_cap1 in every month;
5. does not worsen p05 or stress-net mean versus earliest_ev_cap1 in any month;
6. retains positive episode-rank Spearman in every holdout month for all three
   action horizons; and
7. has a pooled trading-day-cluster bootstrap 95% lower bound above zero.

If this exact branch fails, do not tune the +0.50% execution gate, add a rank
threshold, choose isotonic bins, or alter the fusion using January-March
outcomes. Retire the branch or change the model objective/information set.
