# Selected-HOT position value observability v1.0 — conditional pre-registration

## Status

Request 142 is frozen while request 140B is still running and before any
May-14-through-May-20 economic-opportunity result is inspected.

It may execute only if request 140B passes its frozen promotion gate.

It opens no new date. It reuses:

- fit position paths: 2026-04-30 through 2026-05-06 opened sessions;
- calibration position paths: 2026-05-07 through 2026-05-13 opened sessions;
- evaluation paths: 2026-05-14, 05-15, 05-18, 05-19 and 05-20.

May 21 and later remain sealed.

## Question

After a first-HOT episode has been accepted by the frozen request-140B
ENTRY-opportunity representation, does the evolving causal market state contain
enough information to decide whether one more minute of patience is worthwhile
and whether meaningful multi-minute exit option value remains?

This is an observability bridge, not yet an executable trading policy.

## Entry population

For fit and calibration of the HOLD/EXIT models, use all evaluable first-HOT
position paths so the exit model is not trained on an entry subset selected
in-sample by the opportunity labels.

For the May-14-through-May-20 evaluation diagnostic, report both:

1. all first-HOT paths;
2. the request-140B frozen selected first-HOT subset.

The 140B selection rule and probability threshold may not be changed.

## Position timeline

Entry reference is unchanged: the exact next minute open after the HOT
decision. After entry, evaluate decision states after each completed holding
minute, up to a hard 30-minute research cap.

No missing timestamp is compressed. A missing state is reported rather than
silently shifted.

## Causal position-state features

At each reached decision timestamp use only data then observable:

- the frozen broad minute/state features;
- the frozen trailing one-second path features through that timestamp;
- current price and regular-session clock context;
- minutes held;
- entry-to-current return;
- running post-entry high/low return;
- drawdown from the running post-entry high;
- recovery from the running post-entry low.

Future opens, future returns, later HOT states, oracle exit labels and
post-decision second data are forbidden as features.

## Diagnostic A — one-minute continuation

Using the original entry fill and frozen BASE execution model, define:

hold_advantage_1m_pct = BASE return from exiting at the next exact minute open
minus BASE return from exiting at the current exact open.

Fit a binary classifier for hold_advantage_1m_pct > 0 on fit days and
chronologically Platt-calibrate it on calibration days.

Frozen classifier family:

- HistGradientBoostingClassifier;
- learning_rate 0.05;
- max_iter 180;
- max_leaf_nodes 15;
- min_samples_leaf 75;
- l2_regularization 2.0;
- random_state 20261052.

The semantic HOLD diagnostic is calibrated probability > 0.5. No threshold
search is allowed.

## Diagnostic B — remaining option value

For each current decision state with an executable current exit reference,
compute the best BASE return among later exact observed opens through the same
30-minute cap.

remaining_option_value_pct = best later BASE return - exit-now BASE return.

To remove the trivial advantage of earlier holding minutes, calculate a
fit-only mean baseline separately for every integer held minute and define:

excess_remaining_option_value_pct = remaining option value - fit-minute baseline.

Fit one HistGradientBoostingRegressor on the same causal position-state frame:

- squared-error loss;
- learning_rate 0.05;
- max_iter 180;
- max_leaf_nodes 15;
- min_samples_leaf 75;
- l2_regularization 2.0;
- random_state 20261053.

Winsorize only fit targets at fit-only 0.5% and 99.5%. Use calibration days
only for one additive mean-bias offset. The semantic diagnostic set is adjusted
predicted excess value > 0.

## Required diagnostics

Report separately for all first-HOT paths and the frozen 140B-selected subset:

- valid-entry and decision-state coverage;
- one-minute HOLD classifier ROC AUC;
- realized incremental BASE mean for semantic HOLD rows;
- day-balanced incremental BASE mean for semantic HOLD rows;
- remaining-value global Spearman;
- median eligible same-held-minute Spearman and fraction of positive minute groups;
- realized excess remaining option value for predicted-excess-positive rows;
- day-balanced realized excess value;
- exact one-second feature coverage;
- held-minute distribution and strict date provenance.

## Frozen bridge

Request 142 supports an executable recurrent HOLD/EXIT experiment only if,
on the frozen 140B-selected May-14-through-May-20 subset:

1. valid decision-state coverage is at least 90%;
2. one-minute HOLD ROC AUC is above 0.50;
3. semantic HOLD rows have positive arithmetic incremental BASE mean;
4. semantic HOLD rows have positive day-balanced incremental BASE mean;
5. remaining-value global Spearman is above 0;
6. median eligible same-minute Spearman is above 0;
7. predicted-excess-positive rows have positive day-balanced realized excess value;
8. the signs in conditions 2 through 7 are non-contradictory on at least four
   of five evaluation days where the diagnostic has sufficient support.

If the bridge passes, the next policy must preserve a short-horizon decay
controller and use remaining-option value only as a narrow patience signal.
Direct same-sample Bellman recursion is forbidden because prior v4.0 research
showed self-referential overholding optimism.

If the bridge fails, do not tune the 0.5 boundary, zero excess boundary,
30-minute cap, model capacity or lag geometry on May14-May20.

May 21 and later remain sealed.
