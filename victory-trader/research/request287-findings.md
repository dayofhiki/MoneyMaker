# Request287 — payoff-decomposed bet value

## Result

Request287 completed successfully but failed the bet-value gate.

The experiment changed the decision objective from isolated loss avoidance to
causal expected payoff at the unchanged first causal 2% pullback:

EV = P(win) * E[gain | win] - (1 - P(win)) * E[loss | non-win]

No date, entry timestamp, stop, trailing rule, max hold, or unresolved handling
was changed.

## Conditioned economics

Inside the held-out stage-1 shortlist:

- shortlisted pullback rows: 54
- resolved trades: 36
- all-trade mean: -0.64794%
- positive rate: 30.56%
- severe-loss rate <= -2%: 36.11%

The EV decomposition did not rank outcomes:

- direct-return Spearman: 0.0152
- decomposed-EV Spearman: -0.0541
- EV gain versus direct: -0.0692

The EV>0 rule was too restrictive and not useful:

- resolved EV>0 bets: 3
- EV>0 trade mean: -0.73710%
- EV>0 positive rate: 33.33%
- EV>0 severe-loss rate: 66.67%
- rejected counterfactual mean: -0.63983%
- cash-adjusted decision mean: -0.04513%
- positive decision-mean days: 1/4

## Component signal

The asymmetry from Request286 remained:

- win-probability AUC: 0.3709
- severe-loss probability AUC: 0.6421

So the upside side of the decomposition was unstable while the downside-tail
signal remained materially more informative.

This is why the expected-value formula itself did not help: an EV calculation
is only useful if its component probabilities/magnitudes are learnable. The
current generic positive-return target treats tiny wins and explosive wins as
the same class and is a poor fit for the intended momentum-bet objective.

## EV quartiles

Predicted EV did not monotonically order realized trade return:

- Q1 predicted EV -2.587%, realized -0.022%
- Q2 predicted EV -1.645%, realized -1.534%
- Q3 predicted EV -1.179%, realized -0.576%
- Q4 predicted EV -0.375%, realized -0.459%

That rules out simply changing the EV threshold or position size. The ordering
itself is not yet trustworthy.

## Interpretation

The new decision philosophy is not rejected. Request287 rejected one specific
implementation of it.

The next experiment should stop treating every return above zero as the same
kind of win. Use an asymmetric payoff-state target more appropriate for
small-cap momentum:

- severe downside: return <= -2%
- middle/noise: -2% < return < +2%
- meaningful upside: return >= +2%

Estimate the three probabilities causally, derive the realized payoff attached
to each state from outer-fold training data only, and compute expected bet value
as the probability-weighted payoff. This keeps the focus on large favorable
payoffs rather than classification accuracy.

Do not open May11-20 or June15-19.
