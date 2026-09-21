# Expanded-history position-path-aware continuation v3.5 — pre-registration

## Status

Frozen after corrected v3.4 request 89 and before inspecting any v3.5 output.
Use only September 2025 through March 2026 development data. April 2026 and
later remain sealed.

v3.4 tested whether the then-observable minute state can distinguish executable
`EXIT now` from `HOLD one more minute`. Corrected request 89 completed
successfully, but the frozen bridge failed in January and February.

## v3.4 result that motivates this experiment

On the frozen opportunity-gate stratum:

| Month | AUC | Policy increment | Day-balanced increment | 95% bootstrap lower | Coverage | Frozen pass |
|---|---:|---:|---:|---:|---:|---|
| 2026-01 | 0.5136 | +0.001055% | +0.001788% | -0.010446% | 95.85% | no |
| 2026-02 | 0.5261 | +0.000389% | -0.000737% | -0.012388% | 95.32% | no |
| 2026-03 | 0.5292 | +0.010259% | +0.012311% | +0.002510% | 95.36% | yes |

The signal is weak rather than absent, but v3.4 cannot be composed into a
recurrent policy under the preregistered rule.

## Structural hypothesis

The existing state describes the stock and market well, but the exit model has
little explicit memory of the position itself. A trader deciding whether to
continue holding cares not only about absolute runner state but also about the
path since this specific entry: current profit, favorable/adverse excursion,
how much of a peak has been given back, whether price is rebounding from the
post-entry low, and how recently the post-entry high/low occurred.

v3.5 tests exactly that missing state family. It does not tune a horizon,
threshold, model, fold, action set, execution cost, or promotion rule.

## Frozen data, folds, label and model

Everything below is inherited unchanged from corrected v3.4 request 89:

- exact v3.2 first-anchor panels from request 84;
- frozen 2025 state panels from request 54 and 2026 state panels from request 8;
- strict past-only fit/calibration/evaluation day partitions;
- unchanged v3.3 opportunity model and frozen `P(opportunity) > 0.5` stratum;
- conceptual entry at the exact next-minute open after the frozen anchor;
- at each completed holding minute 1–29, `EXIT now` uses the exact currently
  observable open and `HOLD` uses the exact next-minute open;
- no missing-timestamp fallback;
- BASE future sell friction only for the one-step advantage label;
- the same HistGradientBoostingClassifier, Platt calibration, random seed,
  hyperparameters and strict `P(HOLD) > 0.5` action rule;
- the same 10,000-sample trading-day cluster bootstrap;
- April 2026+ stays sealed.

## The only planned change: entry-relative causal path features

At decision time, only information already observable by that instant may be
used. Add this fixed feature family:

1. current decision-open return versus the exact entry open;
2. post-entry maximum favorable excursion versus entry;
3. post-entry maximum adverse excursion versus entry;
4. drawdown from the post-entry running high;
5. rebound from the post-entry running low;
6. post-entry high-to-low path range;
7. minutes since the post-entry running high;
8. minutes since the post-entry running low;
9. observed-minute fraction since entry, so missing/paused paths are represented
   rather than silently compressed;
10. path efficiency: net entry-relative progress divided by accumulated absolute
    one-minute movement.

Running highs/lows use only completed bars from entry through the currently
observable bar. The decision open itself is known at the decision instant and
is therefore causal. No future bar, future extreme, future label or April data
may enter these features.

## Required diagnostics

Repeat the corrected v3.4 diagnostics unchanged for January, February and March:

- all-anchor and frozen opportunity-gate strata;
- exact-reference coverage and missing reasons;
- AUC, Brier and log loss;
- predicted HOLD rate;
- always-HOLD incremental mean;
- classifier incremental mean and day-balanced mean;
- held-minute and market-phase breakdowns;
- 10,000-sample trading-day cluster bootstrap;
- strict fold provenance.

## Frozen bridge criterion

Proceed to a separately preregistered executable recurrent one-minute
ENTRY→HOLD/EXIT policy only if the opportunity-gate overall stratum satisfies
all five conditions in every one of January, February and March:

1. AUC above 0.5;
2. classifier incremental mean above zero;
3. day-balanced classifier incremental mean above zero;
4. exact-reference coverage at least 90%;
5. trading-day bootstrap 95% lower bound above zero.

If v3.5 fails, do not tune the 0.5 threshold, duration set, correction or feature
subsets against these months. Treat minute-bar position-path state as
insufficient for robust exit timing and move to a new information/target branch
such as finer execution/microstructure observability or a different
state-transition objective. April remains sealed either way.
