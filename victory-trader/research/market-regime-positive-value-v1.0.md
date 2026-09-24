# Request 174 — market-regime positive remaining-value classifier

## Purpose

Request 173 showed that market-wide context improves post-entry ranking:
pooled Spearman rose from 0.0954 to 0.1129 and same-minute median Spearman from
0.0628 to 0.0976. However the regression score's semantic zero boundary did not
produce a stable economic selection.

Request 174 keeps the exact same causal state and directly classifies whether
the fit-minute-baseline-adjusted multi-minute remaining option value is
positive.

This is not the failed one-minute HOLD target.

## Data

No new market dates.

Reuse:
- Request-171 rich-second fit/calibration/evaluation POSITION artifacts;
- Request-163 already-opened market-wide scan.

## Target

After applying fit-only held-minute baselines:

    y = 1 iff excess_remaining_option_value_pct > 0

Future information appears only in this training/evaluation label.

## State

Exactly:
- Request-166 causal POSITION MODEL_FEATURES;
- Request-171 fixed rich-second features;
- Request-173 fixed market-regime features.

No feature selection after June inspection.

## Classifier/calibration

HistGradientBoostingClassifier:
- log loss;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- min leaf 75;
- L2 2.0;
- seed 20261074.

Fit only on the fit partition.

On chronological calibration only, Platt-calibrate the classifier logit.

Semantic selection boundary:

    VALUE-POSITIVE iff P(excess remaining value > 0) > 0.5

Do not tune this threshold on June 8-12.

## Frozen development bridge

Pass only if all hold:

1. market context coverage >=95%;
2. global AUC >=0.55;
3. median eligible same-held-minute AUC >=0.53;
4. semantic selected rate >=10%;
5. selected realized positive-target rate >=58%;
6. selected realized excess mean >0;
7. selected day-balanced excess >0;
8. trading-day bootstrap 95% lower bound for selected excess >0;
9. daily AUC >0.5 and selected realized excess >0 on >=4/5 evaluation days.

Passing is still a development observability bridge, not a profitable policy.
If it passes, compose an actual one-minute-at-a-time recurrent policy on
already-opened data before opening a new fresh block.
