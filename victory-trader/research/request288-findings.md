# Request288 — asymmetric tail-state bet value

## Result

Request288 completed after lowering only the preregistered minimum class support
from 15 to 10. The +2% meaningful-upside definition itself was not changed.

The asymmetric three-state EV gate failed.

Inside the held-out stage-1 shortlist:

- shortlisted pullback rows: 48
- resolved trades: 32
- severe downside rows <= -2%: 12
- middle rows: 15
- meaningful upside rows >= +2%: 5
- baseline trade mean: -0.64113%

Tail-state model:
- downside AUC: 0.5333
- upside AUC: 0.5333
- tail EV Spearman: -0.0363
- EV>0 resolved bets: 1
- that one trade returned -5.7652%
- cash-adjusted decision mean: -0.1341%
- positive decision days: 0/4

So replacing generic win/loss with three payoff regimes did not create a
useful bet-value ordering.

## Important side finding

The direct return regressor evaluated on the exact same held-out shortlist had:

- direct value Spearman: 0.2265

This was substantially better than the tail-state EV (-0.0363).

That matters because direct return prediction is already an expected-value
problem. It does not require the model to separately classify a winner and then
estimate conditional payoff magnitude.

Request287 and Request288 therefore suggest that explicitly decomposing the
bet into noisy probability components is currently making the problem harder,
not easier.

## Interpretation

Keep the value-betting philosophy, but simplify the estimator.

The next diagnostic should inspect the Request288 OOF direct-return score itself:

- how many shortlisted trades have predicted return > 0;
- their realized mean/positive/severe rates;
- realized economics by direct-score quartile;
- whether the high-score bucket is actually better enough to justify a
  train-derived admission/calibration step.

No new market API pull is necessary for this diagnostic because Request288
already saved all OOF rows.

Do not open May11-20 or June15-19.
