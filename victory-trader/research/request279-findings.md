# Request279 — downstream-aligned candidate target

## Result

Request279 passed all code and timing-contract checks but failed the economic
representation gate.

The experiment isolated the candidate target. The old-target and aligned
models used the same pre-HOT feature boundary, model family, May5-8
leave-one-day-out folds, and a fixed 20% training-derived selection fraction.
The held-out day's future candidate distribution was never used to calibrate
its threshold.

### Support

- candidate episodes: 1,602
- frozen-policy resolved-or-cash labels: 1,531 (95.57%)
- pullback entries: 330
- unresolved entered episodes: 71
- pre-HOT features: 38

### Representation

Against the actual frozen downstream economic outcome:

- value Spearman, old target: 0.0893
- value Spearman, downstream-aligned target: 0.1993
- positive-return AUC, old target: 0.5831
- positive-return AUC, downstream-aligned target: 0.6176

Target alignment therefore improved both ranking and positive-return
discrimination.

### Economics

Admit all:
- candidate mean: -0.2666%
- trade mean: -1.5761%
- severe-loss rate: 52.12%

Old-target top-20 selector:
- 406 selected
- candidate mean: -0.0471%
- day-balanced mean: -0.0469%
- trade mean: -1.3286%
- trade positive rate: 17.86%
- severe-loss rate: 48.21%
- positive days: 0/4

Downstream-aligned top-20 selector:
- 333 selected
- candidate mean: -0.0292%
- day-balanced mean: -0.0292%
- trade mean: -1.3629%
- trade positive rate: 23.53%
- severe-loss rate: 47.06%
- positive days: 0/4

Candidate mean improved by only +0.0179 percentage points versus the isolated
old-target baseline, below the preregistered +0.10pp requirement.

## Interpretation

The candidate target mismatch was real, but it was not the whole bottleneck.
A target trained directly on downstream economic outcome sees the problem
better, yet the current 38-feature HOT-time representation cannot separate a
positive trading subset.

The next boundary is representation, not another entry/exit parameter. Keep
the Request279 downstream target and frozen downstream policy, then add only
causal information available by HOT time:

1. multi-minute pre-HOT ticker trajectory and acceleration;
2. same-time cross-sectional market/crowding regime.

Run ablations so any improvement can be attributed to ticker history versus
market context. Do not open May11-20 or June15-19 until a positive candidate
subset appears.
