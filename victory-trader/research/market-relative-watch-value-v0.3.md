# Market-relative WATCH episode value v0.3 — pre-registration

## Status

Frozen after request 116 failed its preregistered gate and before any v0.3
evaluation outcome is inspected.

This experiment changes only the **causal information family**. It retains the
adaptive WATCH-episode target, causal minute-completion timestamps and fixed
model family from v0.2. The retired request-115/116 one-second feature family is
not tuned or reused as the experimental extension.

April 2026 and later remain sealed.

## Research question

Can market-relative price, volume and transaction acceleration from completed
minute bars rank the remaining value of a WATCH episode more robustly than the
current minute baseline?

The hypothesis is that a small-cap surge is not defined only by its own return.
It may be more informative to know whether price velocity, volume intensity and
transaction intensity are accelerating relative to both the ticker's own recent
history and the rest of the eligible market at the same timestamp.

## Frozen chronology

Build sessions from 2026-01-02 through 2026-01-26 inclusive.

- fit: 2026-01-02 through 2026-01-16 trading sessions;
- evaluation: 2026-01-20 through 2026-01-26 trading sessions.

2026-01-19 is a US market holiday. The five evaluation sessions
2026-01-20, 21, 22, 23 and 26 are untouched by this attention-value model
family at preregistration time.

## Population

Use **all** eligible WATCH-entry episodes, not an outcome-dependent sample.

Episode construction is unchanged from v0.2:

- begin only on a non-focus -> WATCH transition;
- WATCH and HOT remain one continuous focus episode;
- end at the first subsequent SCAN/DROP decision or the final focus decision
  of the regular session;
- direct-HOT episodes are excluded;
- WATCH entries after the ticker has already crossed +10% from nominal prior
  close are excluded.

Using all episodes is feasible because this experiment requires no per-ticker
one-second REST calls.

## Target

Keep the v0.2 target unchanged.

For WATCH-entry close P0, over future completed-minute closes that remain inside
the frozen focus episode:

remaining_episode_peak_return_pct =
max(0, max(P_future / P0 - 1) * 100).

Also retain episode duration and runner-in-remaining-episode diagnostics.

## Baseline features

Use the exact request-116 minute baseline:

- attention score and rank;
- return from prior close;
- current completed-minute body/range;
- log minute volume and transaction count;
- one-minute close return;
- one-minute change in prior-close return;
- current/previous minute volume ratio;
- current/previous minute transaction ratio.

## New market-relative acceleration family

All features are computed only from completed minute bars available by the
decision timestamp.

Ticker-temporal features:

- trailing 3-minute close return;
- trailing 5-minute close return;
- one-minute return acceleration versus the previous minute;
- one-minute volume / mean prior-20-minute volume;
- mean last-5-minute volume / mean preceding-20-minute volume;
- one-minute transactions / mean prior-20-minute transactions;
- mean last-5-minute transactions / mean preceding-20-minute transactions;
- current minute range / median prior-20-minute range.

Cross-sectional same-timestamp features:

- percentile rank of one-minute return;
- percentile rank of three-minute return;
- percentile rank of five-minute return;
- percentile rank of one-minute return acceleration;
- percentile rank of one-minute volume acceleration;
- percentile rank of five-minute volume acceleration;
- percentile rank of one-minute transaction acceleration;
- percentile rank of five-minute transaction acceleration;
- percentile rank of range expansion.

Market-regime context at the same timestamp:

- fraction of eligible names with positive one-minute return;
- fraction with positive five-minute return;
- market median one-minute return;
- market median five-minute return;
- market median one-minute volume acceleration;
- market median one-minute transaction acceleration.

No future bar, completed-day statistic, same-day final volume or future runner
count may enter these features.

## Frozen model

Compare the same fixed HistGradientBoostingRegressor twice:

1. baseline: request-116 minute features;
2. extended: baseline plus the market-relative acceleration family above.

Hyperparameters remain:

- learning_rate = 0.05
- max_iter = 120
- max_leaf_nodes = 7
- min_samples_leaf = 20
- l2_regularization = 1.0
- random_state = 17

No feature, model, threshold or hyperparameter search is allowed after the
evaluation sessions are inspected.

## Evaluation

Report pooled and per evaluation session:

- episode rows and ticker count;
- target distribution and runner rate;
- baseline and extended Spearman;
- baseline and extended MAE;
- top prediction quartile realized episode peak return;
- top prediction quartile runner rate;
- incremental Spearman.

Also report feature missingness and the number of full-population eligible
episodes.

## Promotion gate

Carry the market-relative family into a learned attention-score candidate only
if all are true:

1. every evaluation session has at least 100 eligible episode rows;
2. pooled extended Spearman exceeds pooled baseline Spearman;
3. mean per-day incremental Spearman is positive;
4. extended Spearman is at least baseline on at least four of five evaluation
   sessions;
5. pooled extended MAE is no worse than baseline;
6. pooled extended top-quartile realized episode peak return is at least
   baseline;
7. pooled extended top-quartile runner rate is at least baseline.

Passing this gate does not create an entry or trading policy. The next step
would be recurrent WATCH scoring and a finite HOT-budget replay.

If the gate fails, do not tune this feature family on 2026-01-20 through
2026-01-26. The next branch must change the information family or model target
again.
