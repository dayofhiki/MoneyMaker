# Request 199 — causal first-passage directional edge

## Purpose

Request 198 showed that entry fillability and terminal execution are learnable,
while positive realized return and return magnitude are weakly predictable.

Request 199 removes partial-exit, trailing and terminal-fill noise from the
target. It asks the primitive directional question:

**after a causal filled entry, does the upside barrier arrive before the -3%
stop barrier?**

If this is not learnable, current entry-state features do not contain enough
directional edge and further threshold tuning is not justified.

## Data

No new dates.

- train: Request-171 FIT only;
- evaluate: Request-171 chronological CALIBRATION only;
- Request-178 remains sealed.

Execution reference:
- BASE friction;
- 1-second latency;
- five-second entry expiry;
- exact same causal 1-second entry used by Request 197/198.

## Frozen barrier pairs

- tight: stop -3%, upside +5%;
- runner: stop -3%, upside +10%;
- maximum observation window: 30 minutes from entry.

## First-passage label

Starting with the one-second bar whose open filled the entry:

- if only the stop barrier is first touched, status = stop_first;
- if only the upside barrier is first touched, status = take_first;
- if both are first touched inside the same one-second OHLC bar, status =
  ambiguous;
- if neither occurs inside 30 minutes, status = neither;
- entry-unavailable states remain entry_unavailable.

No within-second ordering is invented.

Primary binary target on filled, non-ambiguous states:
- 1 = take_first;
- 0 = stop_first or neither.

Secondary diagnostic uses only decisive take_first/stop_first states.

## Feature ablation

Train two equal-day HGB classifiers per barrier pair:

1. full: current Request-197 feature family;
2. no-rich-second: same family with all RICH_SECOND_FEATURES removed.

No calibration threshold or offset is fitted.

## Calibration report

For each rule and each model report:
- primary AUC / average precision;
- decisive-only AUC / average precision;
- per-day primary AUC;
- take/stop/neither/ambiguous prevalence.

For the full score, report P80/P90/P95 diagnostic groups:
- state count and unique episode count;
- take-first / stop-first / neither rates;
- barrier proxy return, defined only for diagnosis:
  +take_pct for take_first, -3% for stop_first, 0 for neither;
- day-balanced barrier proxy.

The barrier proxy is not an executable P/L claim.

## Interpretation

- full AUC <0.60 with little rich-second uplift: current causal state
  representation lacks usable short-horizon directional edge;
- strong no-rich model: one-second aggregate features are not necessary for the
  directional signal;
- material full-over-no-rich improvement: one-second microstructure contains
  directional information worth developing;
- strong directional classification but negative executable policy later:
  execution/risk harvesting, rather than entry direction, becomes the next
  bottleneck.

No action policy is selected in Request 199.
