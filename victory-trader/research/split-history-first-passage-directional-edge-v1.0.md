# Request 205 — split-history first-passage directional edge

## Purpose

Requests 203 and 204 did not produce a useful economic selected tail. Request
205 tests a different causal source: strictly-prior split / reverse-split
history.

Earlier fixed-15-minute research found that split history did not stabilize the
old EV policy. This request does not reuse or tune that policy. It asks whether
the same frozen split state carries information for the current one-second
first-passage target.

## Data split

No new dates.

- train: Request-171 FIT only;
- evaluate: Request-171 chronological CALIBRATION only;
- Request-178 remains sealed.

First-passage execution/label contract is unchanged:

- BASE friction;
- 1-second entry latency;
- 5-second entry expiry;
- stop barrier -3%;
- tight upside barrier +5%;
- runner upside barrier +10%;
- 30-minute observation cap.

## Strictly-prior split history

For ticker T on trading day D use only split records with:

    D - 730 calendar days <= execution_date < D

Same-day or future split records are forbidden.

A ticker with no split record is a valid zero-history observation.

## Frozen split feature family

Append exactly these eight features, frozen in the earlier split-state research,
to the Request-199 no-rich-second comparator:

1. split_any_730d;
2. reverse_split_count_730d;
3. reverse_split_count_365d;
4. forward_split_count_730d;
5. stock_dividend_count_730d;
6. log1p days since latest reverse split;
7. log latest reverse consolidation factor;
8. log maximum reverse consolidation factor.

No split threshold, reverse-split subgroup, age cutoff, consolidation cutoff,
lookback window, interaction, or feature selection may be chosen from
calibration outcomes.

## Models

For each barrier pair train:

1. comparator: exact Request-199 no-rich-second feature family;
2. split-history: comparator plus the eight frozen split features.

Use the same equal-day HGB classifier family as Requests 199-204.

## Required diagnostics

Report:

- market-wide split query success;
- strict-prior check;
- any-split and reverse-split prevalence;
- comparator and split AUC / AP;
- AUC uplift;
- decisive-only AUC / AP;
- per-day AUC and split-vs-comparator day wins;
- P80/P90/P95 split-score barrier diagnostics;
- all-state and P90 day-balanced barrier proxy.

## Development signal rule

A split-history branch is worth converting into an executable action model only
if at least one barrier pair satisfies all:

1. split AUC >= 0.62;
2. AUC uplift over comparator >= +0.03;
3. split AUC exceeds comparator on at least 6 of 8 calibration days;
4. P90 day-balanced barrier proxy improves by at least +0.50 percentage points
   versus all states;
5. split query completion is 100%;
6. strict-prior coverage is 100%.

This is not a promotion rule. Request-178 remains sealed.

## Failure rule

If Request 205 fails, do not tune split windows, split thresholds,
consolidation cutoffs, barrier sizes, score quantiles, or model capacity on
calibration outcomes.

At that point, stop the sequence of one-off metadata branches and redesign the
learning target / action formulation before testing more context sources.
