# Request294 — local-turn-aware direct entry

## Result

Request294 completed successfully on the frozen May5-8 development window.
The formal gate failed only because the standalone binary 15-second turn AUC
missed its preregistered threshold. Economically, however, this is the first
entry controller in the current crack-selection line to beat immediate entry.

## Signal

The direct fixed-horizon entry-value model improved after adding causal local
turn sequence features:

- Request293 utility Spearman: 0.2247
- Request294 utility Spearman: 0.2614
- Request293 positive-utility AUC: 0.5906
- Request294 positive-utility AUC: 0.6089

The separate 15-second local-turn target was weak but nonzero:

- labeled states: 4,212
- future 15s return mean: +0.0270%
- Spearman: 0.0869
- positive-return AUC: 0.5352

So local sequence information helps the long-horizon entry-value representation
more clearly than it solves a binary "will the next 15 seconds be green?"
classification task.

## Policy economics

97 complete crack-selected episodes:

- entered: 83 = 85.57%
- skipped: 14
- turn-confirmed entries: 80
- strong-value override entries: 3
- mean delay among entries: 30.7s
- median delay: 5s

Cash-adjusted decision utility:

- fixed 2% pullback: +0.801%
- Request293 direct threshold: +1.347%
- immediate entry: +1.732%
- Request294 local-turn-aware: +2.019%

Request294 beat immediate by +0.286pp and Request293 by +0.671pp.

On the same 83 entered episodes:

- mean price improvement vs immediate: +0.319%
- chosen MFE: +5.889%
- immediate MFE: +5.691%
- MFE gain: +0.197pp
- +5% opportunity: 33.73% vs 32.53%
- +10% opportunity: 13.25% vs 12.05%
- +20% opportunity: 3.61% vs 3.61%

This is the first development result where waiting improved price while also
preserving/increasing explosive-upside opportunity and improving total
cash-adjusted decision utility versus immediate entry.

## Important caution

The standalone turn classifier did not meet the frozen AUC threshold:
0.535 < 0.55. Therefore it is premature to claim that the model has learned a
clean universal "bottom reversal detector".

There are two plausible sources of the gain:

1. the 16 local-turn sequence features improved the direct current-entry-value
   model itself;
2. the separate 15-second turn gate improved stopping.

Request294 mixes these effects. They must be ablated before opening fresh dates.

The strong-value override was selected in only one outer fold and produced only
3 entries overall, so the current gain is mostly associated with the
turn-confirmed path rather than the chase override.

## Request295 boundary

Run an exact matched-fold ablation on the same May5-8 states:

A. Request293-style base direct-value threshold.
B. Turn-enriched direct-value threshold with the 16 new local sequence features,
   but no separate turn gate.
C. Full Request294 value + turn gate + optional strong-value override.

For every variant report:
- pooled and per-held-out-day cash-adjusted utility;
- entry rate, delay and price improvement;
- MFE / +10% opportunity preservation;
- number of held-out days beating immediate entry;
- direct-value OOF Spearman/AUC.

The purpose is not more tuning. It is to identify which causal component created
the first positive timing edge.

If one frozen architecture beats immediate entry on a convincing majority of
the four held-out days, freeze it before opening untouched dates for validation.
