# Request285 — fixed post-pullback recheck learnability findings

## Result

Request285 completed successfully and failed the delayed-state signal gate.

The experiment fixed entry at the first causal 2% pullback and exactly one,
two, or three later observed minute states. No future-best entry state was
chosen. Every horizon used the same -2.5% hard stop, 2% trailing drawdown and
10-minute exact max hold.

## Stage-1-shortlist results

First pullback, h0:
- resolved trades: 32
- trade mean: -0.6411%
- positive rate: 31.25%
- severe-loss rate: 37.50%
- value Spearman: 0.1712
- positive AUC: 0.4727
- severe AUC: 0.5375

One observed minute later, h1:
- resolved trades: 28
- trade mean: -1.5166%
- positive rate: 17.86%
- severe-loss rate: 53.57%
- value Spearman: 0.2754
- positive AUC: 0.6174
- severe AUC: 0.5744

Relative to h0:
- value Spearman +0.1042
- positive AUC +0.1447
- severe AUC +0.0369
- trade mean -0.8755 percentage points worse

Two observed minutes later, h2:
- resolved trades: 20
- trade mean: -1.6742%
- positive rate: 15.00%
- severe-loss rate: 60.00%
- value Spearman: -0.0180
- positive AUC: 0.7451
- severe AUC: 0.3542

Three observed minutes later, h3:
- only 12 resolved trades
- trade mean: -1.3615%
- support too small for reliable ranking diagnostics

## Interpretation

One additional minute of information makes positive-outcome classification
materially easier, but entering one full minute later is economically much
worse. This separates two issues:

1. additional post-pullback information is useful;
2. minute-level WAIT is too coarse and sacrifices too much entry price.

Therefore the next experiment should not build a recurrent minute-level WAIT
controller. It should first test whether sub-minute causal information that is
already available at the original pullback decision time can recover the
missing discrimination without delaying entry.

## Next boundary — Request286

At the original first 2% pullback decision time:

- keep the Request282/285 stage-1 top-20 shortlist;
- keep the exact original pullback entry and downstream exit outcome;
- keep the current causal price-path and completed-minute features;
- add only completed one-second aggregates from the trailing 60 seconds,
  ending at decision_t - 1 second;
- compare baseline versus second-enriched leave-one-day-out prediction inside
  the held-out stage-1 shortlist.

Historical one-second aggregates are research state features, not brokerage
ticks or fill guarantees. No second after the decision timestamp may enter the
feature vector.

Do not open May11-20 or June15-19.
