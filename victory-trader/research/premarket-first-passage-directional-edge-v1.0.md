# Request 200 — premarket-context first-passage directional edge

## Purpose

Request 199 found only weak causal directional edge in the current regular-
session price/volume/path representation:

- tight (+5% before -3%) full AUC 0.5816;
- runner (+10% before -3%) full AUC 0.6057;
- adding the richer one-second aggregate family changed AUC by only +0.0051
  and -0.0102 respectively.

Do not tune Request-199 score thresholds.

Request 200 tests one genuinely different causal context axis: **same-day
premarket structure that is fully observable before the 09:30 ET regular
session open**.

The hypothesis is that a regular-session runner's continuation probability may
depend on whether price discovery and participation were already concentrated
in premarket, where the premarket high formed, and whether the current state is
breaking or fading that structure.

## Data and chronology

No new dates.

- train: Request-171 FIT only;
- evaluate: Request-171 chronological CALIBRATION only;
- Request-178 remains sealed.

Use the exact Request-199 one-second causal entry reference:
- BASE friction;
- 1-second latency;
- 5-second entry expiry;
- filled-entry first-passage labels;
- stop barrier -3%;
- tight upside +5%;
- runner upside +10%;
- 30-minute observation cap.

Premarket bars are same-day Massive one-second aggregates with timestamps:

    04:00:00 ET <= t < official regular-session open

No regular-session or future bar may enter a premarket feature.

A ticker-day with no premarket aggregate receives:
- pm_any_activity = 0;
- count/volume features = 0 where naturally defined;
- price-path features = missing.

Missing premarket activity is information and is not a reason to drop a state.

## Frozen premarket feature set

Append exactly these features to the Request-199 no-rich-second comparator:

1. pm_any_activity;
2. pm_log_active_seconds;
3. pm_log_volume;
4. pm_log_transactions;
5. pm_return_pct;
6. pm_range_pct;
7. pm_last_from_high_pct;
8. pm_close_location;
9. pm_high_age_min;
10. pm_last30_return_pct;
11. pm_last30_volume_share;
12. pm_last30_transactions_share;
13. pm_first_vs_previous_close_pct;
14. pm_last_vs_previous_close_pct;
15. pm_high_vs_previous_close_pct;
16. state_vs_pm_high_pct;
17. state_vs_pm_last_pct.

Previous close is reconstructed causally from the current completed-state close
and its already-existing return-from-previous-close feature. No later daily
close is queried.

## Models

For each barrier pair train two equal-day
HistGradientBoostingClassifier models with the same hyperparameters and seed
family as Request 199:

1. comparator: Request-199 feature family excluding RICH_SECOND_FEATURES;
2. premarket: exact comparator plus the 17 frozen premarket features.

No calibration threshold, probability offset, feature selection, or
hyperparameter is fit.

## Calibration report

For tight and runner report:

- premarket activity coverage;
- comparator AUC / AP;
- premarket AUC / AP;
- AUC uplift;
- decisive-only AUC / AP;
- per-day AUC for both models;
- number of days on which premarket AUC exceeds comparator;
- P80/P90/P95 premarket-score diagnostic groups:
  - state count;
  - unique episodes;
  - take-first / stop-first / neither / unavailable rates;
  - barrier-proxy mean and day-balanced mean.

Barrier proxy remains diagnostic only:
- +take_pct for take-first;
- -3% for stop-first;
- 0 for neither.

## Development signal rule

A premarket directional branch is worth turning into an executable action model
only if at least one barrier pair satisfies all:

1. premarket AUC >= 0.62;
2. AUC uplift over comparator >= +0.03;
3. premarket AUC exceeds comparator on at least 6 of 8 calibration days;
4. P90 day-balanced barrier proxy exceeds the corresponding unfiltered
   all-state proxy by at least +0.50 percentage points.

This is not a promotion rule and does not open Request-178.

## Failure rule

If Request 200 fails, do not tune the 04:00 start, feature subset, premarket
window, classifier capacity, barrier size, or score quantile on calibration
outcomes.

Move to another genuinely different causal information axis, with point-in-time
share supply / float-turnover context as the next pre-declared candidate.


## Result — Request 200

Request 200 completed on Request-171 FIT -> chronological CALIBRATION only.
Request-178 remained sealed.

Premarket activity was present for 93.86% of calibration states, so the branch
did not fail for lack of data.

### Tight +5% before -3%

- comparator AUC: 0.5774;
- premarket AUC: 0.5841;
- uplift: +0.0067;
- premarket beat comparator on 4/8 days;
- P90 day-balanced barrier proxy: -1.2145%;
- all-state proxy: -0.8620%;
- P90 uplift: -0.3525pp.

### Runner +10% before -3%

- comparator AUC: 0.6154;
- premarket AUC: 0.6298;
- uplift: +0.0144;
- premarket beat comparator on 5/8 days;
- P90 day-balanced barrier proxy: -1.7341%;
- all-state proxy: -1.1880%;
- P90 uplift: -0.5462pp.

Neither barrier satisfied the frozen development-signal rule.

## Decision

Retire this exact premarket representation without tuning its 04:00 start,
feature subset, score threshold, or barrier pair on calibration outcomes.

Premarket context contains a small amount of directional information but not
enough to produce a useful selected tail. Proceed to the pre-declared next
causal axis: point-in-time share supply / float-turnover context, evaluated on
the same FIT -> CAL first-passage target before any fresh date is opened.
