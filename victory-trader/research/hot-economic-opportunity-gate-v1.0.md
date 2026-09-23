# HOT economic opportunity gate v1.0 — pre-registration

> **Status: superseded before execution.** This request-141 draft was never
> triggered and opened no data. The already-preregistered request 140B branch
> uses the full frozen minute + one-second HOT feature frame and a genuinely
> fresh May14-May20 block, so 140B is authoritative. Nothing in this draft is
> research evidence or permission to reinterpret 140B.


## Status

Request 141 is a development bridge after requests 139-140.

It opens no new date. It uses:

- fit: 2026-04-30, 2026-05-01, 2026-05-04;
- calibration: 2026-05-05, 2026-05-06;
- evaluation: 2026-05-07, 2026-05-08, 2026-05-11, 2026-05-12, 2026-05-13.

All of these sessions were already opened by attention/economic diagnostics.
May 14 and later remain sealed.

## Motivation

Request 139 showed that broad first-HOT entry is BASE-negative at every fixed
holding horizon, while a non-executable 30-minute best-exit ceiling contains a
profitable tail.

Request 140 showed that:

- first and second HOT promotions retain positive mean hindsight opportunity;
- third-and-later promotions are materially weaker;
- the frozen HOT score orders economic opportunity positively but incompletely.

Therefore the next question is whether a small causal episode-context model can
improve economic opportunity ranking beyond the frozen surge-detection score.

## Unit of analysis

Every causal learned_hot_promote transition is an episode.

No future outcome is used to define a promotion.

## Frozen causal features

Only information available at the promotion timestamp is allowed:

- frozen learned HOT score;
- log learned HOT score;
- broad attention score;
- rank inside current HOT set;
- score margin above the current HOT floor;
- promotion index within ticker-day;
- minutes since previous promotion;
- re-promotion flag;
- minutes since regular-session open.

No price outcome, next-bar return, future runner flag, fixed-horizon return or
other future information may enter the feature set.

## Target

The development target is the request-140 non-executable economic opportunity
ceiling: oracle_best_base_pct.

This equals the best frozen BASE round-trip return available from exact observed
opens within 30 minutes after the exact next-minute entry reference.

This is a candidate-value target, not an executable P&L claim.

## Frozen hurdle model

Fit on the three fit days:

1. classifier for P(oracle_best_base_pct > 0);
2. positive-magnitude regressor on positive targets;
3. loss-magnitude regressor on non-positive targets, using absolute loss.

All three models use HistGradientBoosting with:

- learning_rate 0.05;
- max_iter 180;
- max_leaf_nodes 15;
- min_samples_leaf 100;
- l2_regularization 2.0.

The magnitude targets are winsorized only on fit data at 0.5% / 99.5%.

On the two chronological calibration days:

- Platt-calibrate the classifier probability;
- add one mean-bias offset to the positive magnitude;
- add one mean-bias offset to the loss magnitude.

Expected oracle value is P(win) * E(positive magnitude) - (1-P(win)) * E(loss magnitude).

The semantic diagnostic gate is predicted expected oracle value > 0.

No threshold search is allowed.

## Required evaluation diagnostics

On the five request-138 dates report:

- evaluation row count and positive-opportunity base rate;
- frozen HOT-score ROC AUC for oracle-positive classification;
- hurdle classifier ROC AUC, Brier and log loss;
- frozen HOT-score Spearman correlation with oracle return;
- predicted expected-value Spearman correlation with oracle return;
- number/rate of semantic gate-positive rows;
- gate-positive realized oracle mean, median and positive-opportunity rate;
- same metrics by evaluation day.

## Frozen development bridge

Request 141 supports a fresh ENTRY-value validation only if all conditions hold:

1. at least 1,000 evaluable promotion episodes;
2. model opportunity AUC > 0.50;
3. model opportunity AUC strictly exceeds frozen HOT-score AUC;
4. predicted expected-value Spearman strictly exceeds frozen HOT-score Spearman;
5. at least 100 semantic gate-positive episodes and at least 5% selection rate;
6. selected realized oracle mean > 0 and exceeds the full evaluation population;
7. selected oracle-positive rate exceeds the full evaluation population;
8. selected oracle mean > 0 on at least four of five days;
9. at least 20 selected evaluable episodes on every evaluation day.

Passing this bridge does not prove a profitable trading policy. It only
justifies freezing this economic-value formulation for genuinely fresh
May-14-and-later validation.

If it fails, do not tune the zero gate or model capacity on the evaluation
block. Use the diagnostics to decide whether richer causal state is required.

May 14 and later remain sealed.
