# Request281 — conditional candidate value decomposition

## Result

Request281 failed the economic gate. Decomposing the mixed cash/trade target was
informative, but multiplying pullback occurrence probability by conditional
trade value made the selector worse than the direct downstream-value model.

## Frozen support

- candidate rows: 1,602
- no-pullback cash rows: 1,272
- pullback entries: 330
- resolved trades: 259
- unresolved entered rows: 71
- features: 98 (Request280 ticker-history representation)

## Diagnostics

- direct downstream value Spearman: 0.2301
- decomposed expected-value Spearman: 0.2303
- direct positive AUC: 0.6892
- decomposed positive-probability-mass AUC: 0.6524
- pullback occurrence AUC: 0.7934
- conditional trade-value Spearman: 0.2296
- conditional positive-trade AUC: 0.5182
- conditional severe-loss AUC: 0.6073

The model can predict whether a 2% pullback will occur very well. That is not
the same as predicting a desirable trade. Conditional winner classification
is nearly random, while severe-loss discrimination remains modestly useful.

## Economics

Direct downstream top-20 baseline:
- candidate mean: -0.01473%
- trade mean: -0.64794%
- trade positive rate: 30.56%
- severe-loss rate: 36.11%
- positive candidate-mean days: 0/4

Conditional expected-value top-20:
- candidate mean: -0.02292%
- trade mean: -1.57924%
- trade positive rate: 13.04%
- severe-loss rate: 43.48%
- positive candidate-mean days: 0/4

Gain versus direct baseline: -0.00819 percentage points.

## Interpretation

Pullback probability should not be treated as a positive desirability weight.
A stock can be highly predictable to pull back because it is about to fail.
The useful information is asymmetric: both Request281 pre-HOT features and
Request277 pullback-state features are better at identifying severe downside
than at identifying winners.

## Next boundary — Request282

Use the strongest pre-HOT selector from Request280 as the shortlist, then make
a second causal decision when the fixed 2% pullback actually occurs.

- stage 1: pre-HOT ticker-history downstream-value top-20 shortlist;
- stage 2: Request277-style pullback state with strictly completed prior-minute
  context;
- stage 2 action: reject the highest 50% predicted severe-loss risk, using a
  threshold derived only from the outer fold's training pullback states;
- accepted entries continue through the same frozen exit rule.

This matches the intended trader architecture: choose where to look first,
then reassess risk with new information before committing capital.
