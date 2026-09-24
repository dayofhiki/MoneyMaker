# Request 164 — economic signal audit results

Run: `36014765400`  
Artifact: `10814471522`

Request 164 opened no new dates and changed no model or threshold. It reused the
stored Request-163 first-HOT rows and the same June 8-12 fresh block.

## Missing-label diagnosis

Fresh first-HOT rows: 2,145. Economically labeled rows: 1,306 (60.89%).

| reason | rows | share of all | share of missing |
|---|---:|---:|---:|
| labeled | 1,306 | 60.89% | — |
| no exact next-minute entry bar during session | 815 | 38.00% | 97.14% |
| entry exists but no future minute exit bar during session | 16 | 0.75% | 1.91% |
| near-close no future exit | 5 | 0.23% | 0.60% |
| near-close no next-minute entry | 3 | 0.14% | 0.36% |

The old 95% coverage gate is therefore failing almost entirely because the
current economic label insists on an exact next-minute entry bar. This is an
executability/liquidity problem, not primarily a near-close problem.

## Stable causal signals

The strongest univariate directions were learned from the already-opened
development history and then checked on June 8-12 without changing the model.

| signal | favorable direction | development aligned AUC | fresh aligned AUC | fresh positive median | fresh negative median |
|---|---|---:|---:|---:|---:|
| minutes since open | lower / earlier | 0.611 | 0.628 | 50 min | 100 min |
| minute range | higher | 0.593 | 0.596 | 2.16% | 1.58% |
| minute rerank probability | higher | 0.592 | 0.583 | 0.0141 | 0.0080 |
| second rerank probability | higher | 0.592 | 0.582 | 0.0143 | 0.0085 |
| stage-1 hazard probability | higher | 0.585 | 0.577 | 0.0167 | 0.0086 |
| second-scale realized volatility | higher | 0.576 | 0.576 | 0.513% | 0.331% |
| second-scale max run-up | higher | 0.571 | 0.587 | 1.68% | 1.07% |
| one-minute return | higher | 0.569 | 0.571 | 1.96% | 1.54% |
| one-minute return acceleration | higher | 0.569 | 0.572 | 2.04% | 1.62% |
| second-scale max drawdown | lower / deeper pullback | 0.569 | 0.569 | -0.318% | -0.110% |
| attention score | higher | 0.568 | 0.576 | 0.949 | 0.938 |
| attention rank | lower / better rank | 0.563 | 0.574 | 67 | 80 |

The pattern is coherent: economically better opportunities tend to occur
earlier in the session, with stronger current momentum, wider short-term range,
more second-scale movement, stronger learned hazard/rerank scores, and some
intraminute pullback/run-up structure.

## Fixed-horizon BASE net returns

The frozen Request-163 selector was applied unchanged. Entry remains the exact
next-minute open and exits are mechanical at each fixed horizon. BASE execution
friction is unchanged.

| exit after | all HOT mean net | selected mean net | selected positive rate | selected minus all |
|---:|---:|---:|---:|---:|
| 1 min | -1.41% | -1.15% | 16.1% | +0.26pp |
| 2 min | -1.32% | -1.21% | 22.8% | +0.11pp |
| 5 min | -1.20% | -0.87% | 31.6% | +0.34pp |
| 10 min | -1.09% | -0.74% | 35.6% | +0.35pp |
| 15 min | -1.12% | -0.60% | 36.0% | +0.52pp |
| 30 min | -0.89% | -0.83% | 36.2% | +0.06pp |

The selector consistently improves the population but **does not yet produce
positive mean fixed-horizon BASE returns**. Therefore Request 163/164 validates
economic-opportunity ranking, but not an executable profitable trading policy.

## Research consequence

The main bottleneck has moved downstream:

1. keep the validated SCAN -> Focus -> dynamic Active -> HOT attention path;
2. add causal executability/liquidity information to BUY/WAIT/ABSTAIN so
   candidates without a realistically fillable entry are rejected or delayed;
3. learn a recurrent HOLD/EXIT policy rather than using a fixed holding period;
4. evaluate full portfolio P&L with overlapping-position, capital, and execution
   constraints on a new unopened block.

No fixed horizon from this June audit should be selected and called validated:
that would tune on the fresh block.
