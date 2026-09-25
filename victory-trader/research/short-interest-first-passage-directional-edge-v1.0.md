# Request 203 — publication-safe short-interest first-passage edge

## Purpose

Requests 199-202 showed:

- current same-day price/volume/path state has weak directional edge;
- rich one-second aggregates add essentially no directional value;
- premarket context adds only a small amount;
- multi-day history adds modest runner ranking but no positive selected tail;
- point-in-time share supply is the strongest new structural context so far,
  improving the +10%-before--3% runner AUC from about 0.616 to about 0.678, but
  its selected tail is still economically negative.

Request 203 tests a distinct structural source: **published short-interest
positions**. This is not daily short-sale volume. The hypothesis is that the
amount and persistence of outstanding short positioning can change the odds of
a true squeeze-like runner.

## Data split

No new dates.

- train: Request-171 FIT only;
- evaluate: Request-171 chronological CALIBRATION only;
- Request-178 remains sealed.

The first-passage contract is unchanged from Request 199:

- BASE friction;
- 1-second entry latency;
- 5-second entry expiry;
- stop barrier -3%;
- tight upside barrier +5%;
- runner upside barrier +10%;
- 30-minute observation cap.

## Publication-time safety

Use FINRA short-interest records only after their official publication date.

For a state on trading day D:

- the short-interest publication date must be strictly before D;
- a report published on D is forbidden because date-level metadata does not
  establish availability before the intraday decision;
- settlement date alone is never treated as availability time;
- unknown settlement/publication mappings are ignored;
- no future report is backfilled.

The schedule is the official 2026 FINRA publication calendar already encoded in
the repository, including April 15 -> April 24 and April 30 -> May 11.

## Frozen short-interest feature family

Append exactly the existing eight publication-safe features to the Request-199
no-rich-second comparator:

1. short_interest_latest;
2. short_interest_avg_daily_volume_latest;
3. short_interest_days_to_cover_latest;
4. short_interest_change_pct;
5. short_interest_avg_daily_volume_change_pct;
6. short_interest_days_to_cover_change;
7. short_interest_publication_age_days;
8. short_interest_reports_available.

Do not combine Request-201 supply features in this request. This branch must
establish independent value first.

No short-interest threshold, days-to-cover threshold, squeeze cutoff, transform,
interaction, or feature subset may be selected from calibration outcomes.

## Models

For each barrier pair train two equal-day HGB classifiers with the same model
family and hyperparameters as Requests 199-202:

1. comparator: exact Request-199 no-rich-second feature family;
2. short-interest: comparator plus the frozen eight short-interest features.

No score threshold or probability calibration is fitted.

## Required diagnostics

Report:

- market-wide short-interest query completion;
- latest-record coverage;
- two-record coverage;
- days-to-cover coverage;
- publication-time safety check;
- comparator and short-interest AUC / AP;
- AUC uplift;
- decisive-only AUC / AP;
- per-day AUC and short-interest-vs-comparator day wins;
- P80/P90/P95 short-interest-score barrier diagnostics;
- all-state and P90 day-balanced barrier proxy.

## Development signal rule

A short-interest branch is worth converting into an executable action model only
if at least one barrier pair satisfies all:

1. short-interest AUC >= 0.62;
2. AUC uplift over comparator >= +0.03;
3. short-interest AUC exceeds comparator on at least 6 of 8 calibration days;
4. P90 day-balanced barrier proxy improves by at least +0.50 percentage points
   versus all states;
5. latest-record coverage >= 80%;
6. every usable record satisfies official publication_date < trading_day.

This is not a promotion rule. Request-178 remains sealed.

## Failure rule

If Request 203 fails, do not tune short-interest thresholds, publication-age
cutoffs, model capacity, barrier sizes, feature subsets, or score quantiles on
calibration outcomes.

Proceed to another genuinely different causal source. Only an independently
successful branch may later be combined with Request-201 supply in a separately
pre-registered interaction experiment.
