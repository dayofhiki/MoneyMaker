# First-HOT economic opportunity learnability v1.1 — pre-registration

## Status

## Administrative request ID note

A separate no-new-date HOT-promotion diagnostic was concurrently preregistered
as request 140 first. This fresh learnability branch is therefore designated
**request 140B** for the research record. The already-triggered workflow
metadata still emits integer request_id 140; that is an administrative ID
collision only. The v1.1 methodology, gates and fresh dates below were frozen
before any 140B result was inspected.

Frozen after request 139 established that the promoted HOT population contains
real but sparse economic opportunity, and before inspecting any request-140
output.

Request 139 found 2,091 first-HOT episodes on 2026-05-07 through 2026-05-13.
Blind fixed 1/2/5/10/15/30-minute BASE returns were all negative, while the
non-executable best observed BASE exit within 30 minutes averaged +0.339% and
was positive in 40.70% of evaluable episodes. The median oracle result remained
negative.

The next question is therefore not which fixed holding time to use. It is
whether strictly causal state at first HOT can distinguish the minority of HOT
episodes that contain enough subsequent movement to overcome the frozen BASE
friction model.

## Chronology

The request-138 attention hierarchy is frozen.

Entry-opportunity model development uses only already-opened dates:

- fit: 2026-04-30, 2026-05-01, 2026-05-04, 2026-05-05, 2026-05-06;
- calibration: 2026-05-07, 2026-05-08, 2026-05-11, 2026-05-12, 2026-05-13.

Fresh evaluation opens only:

- 2026-05-14;
- 2026-05-15;
- 2026-05-18;
- 2026-05-19;
- 2026-05-20.

May 21 and later remain sealed.

The upstream market-wide hazard fit remains through 2026-01-16 and the
one-second HOT reranker fit remains 2026-01-20 through 2026-02-09.

## Frozen first-HOT episode

One episode is one ticker-day.

Use only the first chronological state in which the frozen hierarchy assigns
that ticker to HOT. No later HOT re-entry may create another training or
evaluation episode.

Economic entry reference remains the exact next minute-bar open after the HOT
decision. No missing reference is shifted to another timestamp.

## Economic opportunity target

For every valid first-HOT entry reference, inspect observed positive minute
opens strictly after entry and no later than 30 minutes after entry.

Apply the unchanged BASE execution model to the entry and every candidate exit.

Define:

    oracle_best_base_pct =
        maximum BASE round-trip return among observed exits through 30 minutes

and

    opportunity_positive =
        1 if oracle_best_base_pct > 0, else 0

This is a hindsight label, not an executable exit policy. It is permitted only
as a supervised target for testing whether current causal state contains
information about future economic opportunity.

If no future observed exit exists inside the window, the episode is unlabeled
rather than assigned zero.

The 30-minute cap is inherited as a research/safety horizon and is not a claim
that the final trader should hold for 30 minutes.

## Frozen causal feature set

Use the exact current HOT-reranker causal feature frame plus economic context
known at the decision instant:

- all frozen minute + state + one-second reranker features;
- frozen minute-rerank probability;
- frozen second-rerank probability;
- log current price;
- minutes since regular-session open;
- minutes remaining to regular-session close;
- current rank by second-rerank probability inside active observation;
- whether the ticker is currently in the desired top-20 rather than retained
  only as transport state.

Absolute price is included because the frozen execution model has minimum
cent-based spread costs, making friction strongly price-dependent.

No future return, crossing outcome, later HOT state, oracle label, fixed-horizon
return or post-decision observation may enter the feature matrix.

## Frozen models

### Opportunity classifier

Fit one HistGradientBoostingClassifier on fit days:

- learning_rate = 0.05;
- max_iter = 160;
- max_leaf_nodes = 15;
- min_samples_leaf = 60;
- l2_regularization = 2.0;
- random_state = 20261050.

Target: opportunity_positive.

No class weights or evaluation-driven threshold tuning.

### Opportunity-value regressor

Fit one HistGradientBoostingRegressor on the same causal feature frame:

- squared-error loss;
- learning_rate = 0.05;
- max_iter = 180;
- max_leaf_nodes = 15;
- min_samples_leaf = 60;
- l2_regularization = 2.0;
- random_state = 20261051.

Target: oracle_best_base_pct.

Winsorize only fit targets at their fit-only 0.5th and 99.5th percentiles.
On calibration days, calculate one additive mean-bias offset:

    offset = mean(actual oracle value - raw predicted value)

Freeze that offset before evaluation.

## Fixed selection diagnostic

The classifier is primarily a learnability test, not yet an entry policy.

On calibration rows, freeze the 75th percentile of predicted opportunity
probability. On evaluation rows, call an episode "selected" when its probability
is at or above that frozen score boundary.

The percentile boundary uses no calibration outcome label and may not be
changed after evaluation.

## Required diagnostics

Report pooled and per-day:

- first-HOT episodes and economic-label coverage;
- opportunity-positive base rate;
- classifier ROC AUC, average precision and Brier score;
- regressor Spearman correlation with oracle_best_base_pct;
- frozen calibration 75th-percentile probability threshold;
- selected count/rate;
- all-HOT versus selected oracle BASE mean;
- all-HOT versus selected opportunity-positive rate;
- upstream second-data coverage and HOT occupancy;
- strict date provenance.

## Frozen promotion gate

Request 140 passes only if all conditions hold:

1. every fresh evaluation day has at least 100 labeled first-HOT episodes;
2. economic-label coverage is at least 95% on every fresh day;
3. pooled classifier ROC AUC >= 0.55;
4. classifier ROC AUC > 0.50 on at least four of five fresh days;
5. pooled opportunity-value Spearman >= 0.05;
6. opportunity-value Spearman > 0 on at least four of five fresh days;
7. the frozen top-quartile selected set has positive mean oracle BASE value;
8. selected mean oracle BASE value strictly exceeds all-HOT mean;
9. selected opportunity-positive rate exceeds the all-HOT rate by at least
   five percentage points;
10. selected mean oracle BASE value is non-lower than the daily all-HOT mean on
    at least four of five fresh days.

These gates test useful causal ordering, not live profitability.

## Decision rule

If request 140 passes, freeze the economic-opportunity representation and move
to an executable ENTRY/ABSTAIN experiment. That next experiment must evaluate
realized returns under a causal exit controller rather than oracle exits.

If request 140 fails, do not lower the AUC/correlation/selection gates on the
fresh block. Diagnose whether failure comes from insufficient first-HOT state,
too-short development history, or economic opportunity being largely
unpredictable from the currently available information.

No fixed holding horizon may be selected from request-139 or request-140
outcomes.

May 21 and later remain sealed.

## Result — request 140B

Authoritative fresh run: `35829190438`, completed successfully. Evaluation used
only 2026-05-14, 05-15, 05-18, 05-19 and 05-20; May 21 and later remained
sealed.

The economic-opportunity signal replicated strongly on 1,881 labeled first-HOT
episodes. Pooled classifier ROC AUC was 0.6334 and exceeded random on all five
days (daily AUC 0.6126 to 0.6464). The opportunity-value regressor achieved
Spearman 0.2704 and was positive on all five days (0.1733 to 0.3241).

The calibration-frozen top-quartile score boundary selected 503 / 1,881
labeled episodes (26.74%). Their hindsight 30-minute oracle BASE mean was
+1.255% versus +0.452% across all first-HOT episodes, and their cost-positive
opportunity rate was 53.28% versus 42.42% overall. Selected oracle mean exceeded
the daily all-HOT mean on all five fresh sessions.

The formal preregistered promotion gate remained false for one non-predictive
reason: 2026-05-15 economic-label coverage was 94.59%, below the frozen 95%
floor by 0.41 percentage points. The other four days had 97.27% to 98.10%
coverage. All predictive ordering and selection-effect conditions passed.
This result is therefore not retroactively promoted, but it establishes strong
fresh evidence that causal first-HOT state contains learnable economic-value
information.

### Decision

Do not weaken the 95% coverage gate or relabel request 140B as a pass. Diagnose
the small May-15 label-coverage shortfall without tuning predictive models on
this block. The active research frontier nevertheless moves to a preregistered
executable policy experiment: combine the frozen opportunity representation
with causal BUY/WAIT/ABSTAIN and recurrent HOLD/EXIT, while keeping May 21 and
later sealed until that policy is frozen.
