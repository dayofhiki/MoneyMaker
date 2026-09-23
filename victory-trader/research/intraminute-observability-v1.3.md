# Intraminute observability v1.3 — diagnostic pre-registration

## Status

Diagnostic-only follow-up to request 132. It uses only the already-opened
2026-04-01, 04-02, 04-06, 04-07 and 04-08 sessions. April 9 and later are not
opened by this diagnostic.

## Question

For first +10% crossings that have no exact completed minute immediately before
the crossing minute, how much observable one-second activity exists inside the
crossing minute before the +10% threshold is reached?

Request 132 showed that only 557 of 1,089 crossings had an eligible exact prior
minute row. The focus ranker captured 538 of those 557. This diagnostic tests
whether the remaining completed-minute blind spots contain intraminute temporal
structure that a future event-driven scanner could observe.

## Frozen diagnostic

- Rebuild the same point-in-time common-stock minute scan on the five already
  opened sessions.
- Identify first +10% crossing rows.
- Classify each crossing by whether the immediately preceding completed minute
  exists, whether it is the ticker's first observed minute, or whether the
  previous activity is separated by a multi-minute gap.
- Only for crossings without an exact prior minute, fetch historical one-second
  aggregates for the crossing ticker/day.
- Restrict analysis to seconds inside the crossing minute.
- Use the same nominal prior close and +10% definition as the minute replay.
- Measure the first active second, the first second close at or above +10%, the
  number of active seconds before threshold, wall-clock time from first activity
  to threshold, and peak pre-threshold return.
- No predictive feature, threshold, selector or trading rule is fitted.

## Outputs

Report:

- total crossings and exact-prior-minute observability rate;
- blind crossing count and minute-gap categories;
- one-second data coverage for blind crossings;
- fraction whose first active second is already at/above +10%;
- fraction with any pre-threshold active second;
- descriptive rates with at least 5, 10 and 30 active pre-threshold seconds;
- distributions of active seconds and wall-clock seconds before threshold;
- first-active and pre-threshold return distributions;
- pooled, per-day and per-gap-category summaries.

## Interpretation

If a substantial fraction of minute-blind crossings contains pre-threshold
second-level activity, the next scanner milestone should explicitly move broad
market attention from completed-minute cadence toward event-driven or
multi-resolution updates. If most blind crossings gap above +10% on their first
active second, those cases are not recoverable by faster inference alone and
must instead be treated as unobservable jump risk.

This diagnostic does not itself justify a trading entry rule. It only measures
the information that a continuous scanner could have had before the current
minute-level replay declares the crossing.
