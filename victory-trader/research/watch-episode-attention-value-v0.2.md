# Watch-episode attention value v0.2 — pre-registration

## Status

Frozen after request 115 failed its preregistered fixed-three-minute gate and
before any v0.2 evaluation outcomes are inspected.

This branch changes the **target formulation**, not the failed request-115
three-minute horizon, feature list, thresholds or hyperparameters. It remains a
development-only attention experiment. April 2026 and later remain sealed.

## Motivation

Request 115 found directionally positive but unstable incremental information
from the just-completed 60-second path. Pooled Spearman, MAE and top-quartile
outcomes improved, but one of two evaluation days had lower rank correlation,
so the gate failed.

The final trader should not be defined by a fixed three-minute forecast. The
next question is therefore whether the same causal WATCH-state information can
rank **remaining attention value over the natural lifetime of a WATCH/HOT
episode**.

## Frozen development chronology

Build sessions from 2026-01-02 through 2026-01-16 inclusive.

- fit: 2026-01-02 through 2026-01-09 trading sessions;
- evaluation: 2026-01-12 through 2026-01-16 trading sessions.

The five evaluation sessions are fresh to this attention-model family at
preregistration time. They are development evaluation, not final sealed
validation.

## Attention episode

Use the unchanged Phase-2F causal minute-completion replay and frozen
AttentionConfig.

A candidate episode begins only when a ticker transitions from a non-focus
state into **WATCH**. A ticker that enters HOT directly begins a separate
already-HOT episode and is not a candidate for this WATCH -> HOT diagnostic.

Once a candidate episode begins, WATCH and HOT are treated as one continuous
focus episode until the first subsequent SCAN or DROP state. If no such state
appears before session end, the episode ends at the last focus decision.

Only episodes whose WATCH entry occurs before the ticker has already crossed
the existing +10% prior-close runner threshold are eligible.

## Frozen sampling

For each trading session:

1. enumerate eligible candidate ticker-days without inspecting episode outcomes;
2. sort ticker-days by SHA256(`trading_day|ticker`);
3. select the first 24 ticker-days;
4. retain at most the first two eligible WATCH-entry episodes for each selected
   ticker-day;
5. fetch historical one-second aggregates once per selected ticker-day.

This is a new target formulation with a larger resource-bounded sample, not a
retry of the request-115 fixed-three-minute gate.

## Causal features

At each WATCH-entry decision, use the same frozen feature families as request
115.

Minute baseline:

- attention score and rank;
- prior-close return;
- completed-minute body/range;
- completed-minute log volume and transaction count;
- one-minute return and return acceleration;
- volume and transaction ratios to the previous completed minute.

One-second extension:

- active-second counts and last-activity age;
- 10s and 30s return shape/acceleration;
- realized second-close volatility and positive-return fraction;
- intra-minute max drawdown/run-up;
- last-10s volume/transaction concentration;
- last-10s versus previous-10s volume/transaction burst.

For a WATCH entry at `t`, a one-second bar is a feature only when its full
one-second interval completed by `t`.

## Primary target: remaining episode peak return

For entry close `P0`, inspect completed-minute closes strictly after the WATCH
entry while the frozen focus episode remains active.

Define:

`remaining_episode_peak_return_pct = max(0, max(P_future / P0 - 1) * 100)`.

If the episode ends at a SCAN/DROP decision, that non-focus decision timestamp
is excluded from the target window. If the episode remains focused into session
end, the last focus timestamp is included.

This target has no fixed forecast horizon. Its horizon is determined by the
causal state episode produced by the frozen attention runtime.

Secondary diagnostics:

- focus-episode duration;
- whether a +10% prior-close runner crossing occurs during the remaining
  episode.

## Frozen model

Use the exact request-115 HistGradientBoostingRegressor configuration for both
models:

- learning_rate = 0.05
- max_iter = 120
- max_leaf_nodes = 7
- min_samples_leaf = 20
- l2_regularization = 1.0
- random_state = 17

Compare:

1. minute-only baseline;
2. minute + frozen one-second extension.

No feature, model or hyperparameter search is allowed after evaluation.

## Evaluation

Report pooled and per evaluation day:

- rows and ticker count;
- episode-duration distribution;
- target mean/median/p75/p90;
- runner-in-episode rate;
- baseline and extended Spearman;
- baseline and extended MAE;
- realized episode peak return in each model's top prediction quartile;
- incremental Spearman.

Also report one-second data coverage, cache/network request counts and sampled
ticker-days.

## Information-value gate

Carry the one-second path family forward into a learned attention-score
candidate only if all are true:

1. every evaluation session has at least 15 eligible episode rows;
2. pooled extended Spearman exceeds pooled baseline Spearman;
3. mean per-day extended Spearman is at least the mean per-day baseline
   Spearman;
4. extended Spearman is at least baseline on at least three of the five
   evaluation sessions;
5. pooled extended MAE is no worse than baseline;
6. pooled extended top-quartile realized episode peak return is at least
   baseline.

Passing this gate still does not create a BUY policy or establish profitability.

If it fails, retire this frozen one-second aggregate feature family as the next
attention-model branch and move to a genuinely new causal information family,
such as market-relative price/volume/transaction acceleration, without tuning
v0.2 against the failed dates.
