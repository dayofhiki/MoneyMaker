# Request 204 — recurrent watch-to-entry controller v1.0

## Purpose

The previous entry experiments implicitly treated "good candidate discovered"
too much like "buy candidate now." Request 204 changes the control problem.

A ticker entering HOT begins a **watch episode**, not a trade. The controller
continues observing causal states and chooses one of three actions at each
checkpoint:

- ENTER_NOW;
- WAIT;
- DROP.

This first experiment isolates entry timing. Once an entry is admitted, the
existing frozen runner risk policy is used unchanged so that any improvement can
be attributed to the watch-to-entry controller rather than a simultaneous exit
redesign.

## Data split

No new dates.

- training/evaluation source: Request-171 only;
- model training: Request-171 FIT only;
- evaluation: Request-171 chronological CALIBRATION only;
- Request-178 fresh dates remain sealed.

Decision checkpoints use the already-built causal shadow states at minutes
1 through 10 after HOT. Order execution remains one-second causal:

- BASE friction;
- 1-second latency;
- 5-second entry expiry;
- an expired/unavailable order does not create a position and observation may
  continue.

After an admitted entry, use the unchanged runner rule from Request 197:

- hard stop -3%;
- +10% partial take of one half;
- 7% proportional trailing retracement on the remainder;
- 30-minute research cap.

The 30-minute cap remains a research safety cap, not the intended final live
holding rule.

## Core idea

For every causal watch state, define two training values from the frozen runner
replay:

1. **enter-now value**: realized BASE net return if the runner rule is entered
   from this exact state;
2. **wait value**: the best realized BASE net return among later observable
   watch states in the same episode.

The second quantity is a hindsight training label only. It is never available to
the executable controller. It teaches the model the difference between:

- "this is already the best moment available";
- "the ticker may still be good, but waiting offered a better entry";
- "nothing profitable remained in the observed watch window."

At evaluation time only causal features and model predictions are used.

## Frozen sequential action rule

No calibration quantile or hand-picked pullback threshold is allowed.

At each evaluation state:

1. predict ENTER_NOW value;
2. predict WAIT value when a later checkpoint exists;
3. if both predicted values are <= 0: DROP;
4. else if ENTER_NOW value > 0 and ENTER_NOW value >= WAIT value: submit entry;
5. if the submitted order is unavailable before expiry: remain flat and continue;
6. otherwise the entry is admitted and the watch episode ends;
7. in all other cases: WAIT;
8. at the final minute-10 checkpoint there is no WAIT action. ENTER only when
   predicted ENTER_NOW value > 0, otherwise DROP.

This permits both behaviors the user described:
- wait through an overextended move and enter after a better setup forms;
- enter immediately when the model believes waiting is worse than buying now.

## Frozen causal path features

The existing Request-197 causal feature family remains the base. Append exactly
these watch-episode features:

1. watch_location_in_running_range;
2. watch_peak_age_minutes;
3. watch_trough_age_minutes;
4. watch_drawdown_change_1m;
5. watch_recovery_change_1m;
6. watch_attention_change_1m;
7. watch_attention_change_from_first;
8. watch_log_volume_change_1m;
9. watch_log_volume_change_from_first;
10. watch_price_change_from_first_pct.

Definitions use only the current and earlier states of the same
(trading_day, ticker, hot_t) episode.

No feature is named or thresholded as a mandatory "pullback", "breakout" or
"chase" rule. The model must learn those behaviors from path evolution.

## Models

Train two equal-day HistGradientBoostingRegressor heads:

- enter-now head;
- wait-value head.

Use:
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=15;
- min_samples_leaf=75;
- l2_regularization=2.0.

No calibration-data offset, threshold, quantile, or hyperparameter is fitted.

Report two controllers on calibration:
1. BASE_FEATURE_CONTROLLER using the existing Request-197 feature family only;
2. PATH_CONTROLLER using the same family plus the ten frozen watch-path features.

This distinguishes the value of the recurrent action structure from the
additional path description.

## Baselines and diagnostics

Report:

- earliest-admitted baseline: enter at the first 1-10m checkpoint with a legal
  fill;
- PATH_CONTROLLER;
- BASE_FEATURE_CONTROLLER;
- hindsight best admissible entry per episode as an oracle ceiling, clearly
  labeled non-executable.

For each executable policy report:

- candidate episodes;
- ENTER / DROP / no-action counts;
- mean chosen entry minute;
- entry-unavailable retries;
- admitted trades;
- closed coverage;
- mean BASE net return;
- day-balanced BASE net return;
- positive trade rate;
- average winner / loser;
- payoff ratio;
- severe-loss rate;
- positive days;
- day-balanced bootstrap interval;
- improvement versus earliest-admitted baseline on matched episodes.

Also report action behavior by chosen minute so we can tell whether the model
actually learned to wait sometimes rather than simply recreating a fixed delay.

## Development success rule

PATH_CONTROLLER passes this development stage only if all hold on calibration:

1. at least 30 admitted trades;
2. closed coverage >= 90%;
3. day-balanced BASE net return > 0;
4. at least 6 of 8 calibration days have positive mean return;
5. day-balanced bootstrap 95% lower bound > 0;
6. matched-episode day-balanced improvement versus earliest-admitted baseline
   >= +0.50 percentage points;
7. severe-loss rate is not worse than earliest-admitted baseline;
8. chosen entries are not concentrated entirely at one fixed minute;
9. PATH_CONTROLLER day-balanced return is at least as high as
   BASE_FEATURE_CONTROLLER.

Passing Request 204 still does not open Request-178 automatically. It only
justifies a separately pre-registered recurrent post-entry HOLD/EXIT integration
before fresh evaluation.

## Failure rule

If Request 204 fails, do not tune minute checkpoints, zero boundary, score
quantiles, pullback depths, model capacity or the 10-minute watch cap on
calibration outcomes.

Use the failure diagnostics to determine whether the problem is:
- value prediction;
- insufficient causal path description;
- watch window too impoverished;
- or the candidate population itself.
