# Request280 — causal pre-HOT context ablation

## Result

Request280 completed successfully after an implementation-only performance
optimization that restricted preprocessing to the already-open May5-8 dates
and reused one annotated minute frame. The research contract was unchanged.

The representation economic gate failed, but the ablation identified a useful
direction: ticker-specific pre-HOT trajectory helps; broad same-time market
context does not.

## Frozen contract

- May5-8 leave-one-day-out only
- downstream-aligned Request279 target
- top-20% threshold derived from each fold's training days only
- first causal 2% pullback
- -2.5% hard stop
- 2.0% trailing drawdown
- 10-minute exact max hold
- missing exact deadline remains unresolved
- no May11-20 or June15-19

## Representations

Base:
- 38 features
- candidate mean -0.02729%
- trade mean -1.35179%
- trade positive rate 21.88%
- severe-loss rate 43.75%
- value Spearman 0.2259
- positive AUC 0.6731
- positive candidate-mean days 0/4

Base + exact 1/3/5-minute ticker trajectory:
- 98 features
- candidate mean -0.01473%
- trade mean -0.64794%
- trade positive rate 30.56%
- severe-loss rate 36.11%
- value Spearman 0.2301
- positive AUC 0.6892
- positive candidate-mean days 0/4

Ticker trajectory + same-time market context:
- 109 features
- candidate mean -0.02754%
- trade mean -1.36741%
- trade positive rate 18.75%
- severe-loss rate 46.88%
- value Spearman 0.2102
- positive AUC 0.6423
- positive candidate-mean days 0/4

Ticker trajectory improved candidate mean by +0.01257 percentage points versus
the base representation and roughly halved the selected-trade loss. Broad
market context reversed the improvement.

## Coverage note

Exact pre-HOT ticker history is sparse in the source scan:
- 1-minute history: about 55% coverage
- 3-minute history: about 33%
- 5-minute history: about 24%

Despite that sparsity, the ticker-history model improved economics, suggesting
that local trajectory contains real information. The broad market aggregate
features do not justify their added complexity in this form.

## New bottleneck

The downstream candidate target mixes qualitatively different outcomes:

- total candidates: 1,602
- candidates with a 2% pullback entry: 330
- resolved entered trades: 259
- unresolved entered trades: 71
- no-pullback cash candidates: 1,272

Therefore roughly four-fifths of candidate targets are cash zeros. A single
regressor can improve its loss by identifying candidates that simply never
trade, rather than by identifying profitable trades.

## Next boundary — Request281

Keep the ticker-history representation and frozen trading policy, but decompose
the candidate outcome into causal pre-HOT heads:

1. probability that the fixed 2% pullback entry occurs;
2. conditional trade return given an entered, resolved episode;
3. conditional probability of a positive trade;
4. conditional probability of a severe loss.

Form candidate expected value from pullback probability times conditional
trade value, and compare it with the direct downstream-value model under the
same leave-one-day-out and train-derived top-20% selection contract.

This tests whether the remaining failure is target-mixture collapse rather
than another missing price-action or exit parameter.
