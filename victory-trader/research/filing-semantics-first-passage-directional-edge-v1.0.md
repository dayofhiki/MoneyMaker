# Request 204 — filing-semantics first-passage directional edge

## Purpose

Request 203 showed that publication-safe FINRA short-interest positions do not
improve the current risk-compatible first-passage edge.

Request 204 tests a genuinely different causal source: **strictly-prior SEC 8-K
filing semantics**.

Earlier research already showed that coarse 8-K metadata did not stabilize an
older fixed-15-minute EV policy. This request does not retune that policy. It
asks a different, narrower question under the current one-second execution
contract:

**Do recent equity-supply, debt, operating-catalyst, or adverse corporate
filings help distinguish a +5%/+10% move that arrives before the -3% stop?**

## Data split

No new dates.

- train: Request-171 FIT only;
- evaluate: Request-171 chronological CALIBRATION only;
- Request-178 remains sealed.

Execution / label contract is exactly Request 199:

- BASE friction;
- 1-second entry latency;
- 5-second entry expiry;
- stop barrier -3%;
- tight upside barrier +5%;
- runner upside barrier +10%;
- 30-minute observation cap.

## Point-in-time rule

For trading day D include only 8-K disclosures with:

    D - 30 calendar days <= filing_date < D

Same-day filings are forbidden because date-only filing metadata does not prove
availability before the intraday decision.

The market-wide retrieval implementation must treat a ticker with no matching
filing as a valid zero-filing observation, not a failed query.

## Frozen semantic feature family

Use exactly these eleven transforms, defined before this run and already present
in the repository's earlier filing-semantics research:

1. log1p equity-supply accessions, 30d;
2. log1p equity-supply accessions, 7d;
3. log1p debt accessions, 30d;
4. log1p debt accessions, 7d;
5. log1p operating-catalyst accessions, 30d;
6. log1p operating-catalyst accessions, 7d;
7. log1p adverse accessions, 30d;
8. log1p adverse accessions, 7d;
9. log1p total semantic accessions, 30d;
10. log1p age of latest semantic filing;
11. (equity_supply_30d - operating_30d) /
    (equity_supply_30d + operating_30d + 1).

The semantic buckets remain frozen:

- equity supply: public offering, private placement, underwriting agreement,
  acquisition-consideration shares;
- debt: secondary category debt_activity;
- operating catalyst: clinical trial results, preliminary/quarterly results,
  guidance update, partnership/collaboration, licensing agreement, strategic
  initiative;
- adverse: material litigation and executive/CFO/director departures.

No category, keyword, window, or interaction may be changed after calibration
outcomes are seen.

## Models

For each barrier pair train:

1. comparator: exact Request-199 no-rich-second feature family;
2. filing-semantic: comparator plus the eleven frozen semantic features.

Use the same equal-day HGB classifier family as Requests 199-203. No action
threshold or probability calibration is fitted.

## Required diagnostics

Report:

- market-wide filing query completion;
- strict-prior check;
- 30d semantic-filing prevalence;
- equity-supply / debt / operating / adverse prevalence;
- comparator and semantic AUC / AP;
- AUC uplift;
- decisive-only AUC / AP;
- per-day AUC and semantic-vs-comparator day wins;
- P80/P90/P95 semantic-score barrier diagnostics;
- all-state and P90 day-balanced barrier proxy.

## Development signal rule

A filing-semantic branch is worth converting into an executable action model
only if at least one barrier pair satisfies all:

1. semantic AUC >= 0.62;
2. AUC uplift over comparator >= +0.03;
3. semantic AUC exceeds comparator on at least 6 of 8 calibration days;
4. P90 day-balanced barrier proxy improves by at least +0.50 percentage points
   versus all states;
5. query completion is 100%;
6. strict-prior coverage is 100%.

This is not a promotion rule. Request-178 remains sealed.

## Failure rule

If Request 204 fails, do not tune filing windows, semantic buckets, category
thresholds, model capacity, barrier sizes, feature subsets, or score quantiles
on calibration outcomes.

Move to another genuinely different causal information source rather than
recombining failed branches.


## Result — Request 204

Request 204 completed successfully on Request-171 FIT -> chronological
CALIBRATION only. Request-178 remained sealed.

Calibration filing coverage:
- completed query coverage: 100%;
- strict-prior coverage: 100%;
- any frozen semantic filing in prior 30d: 24.94%;
- equity-supply filing prevalence: 10.82%;
- debt: 6.12%;
- operating catalyst: 8.00%;
- adverse: 8.24%.

### Tight +5% before -3%

- comparator AUC: 0.5765;
- filing-semantic AUC: 0.5723;
- uplift: -0.0042;
- semantic model beat comparator on 6/8 days;
- decisive-only AUC: 0.4970;
- P90 day-balanced barrier proxy: -1.1923%;
- all-state proxy: -0.8620%;
- P90 uplift: -0.3303pp.

### Runner +10% before -3%

- comparator AUC: 0.6159;
- filing-semantic AUC: 0.6327;
- uplift: +0.0168;
- semantic model beat comparator on 7/8 days;
- decisive-only AUC: 0.5575;
- P90 day-balanced barrier proxy: -1.2054%;
- all-state proxy: -1.1880%;
- P90 uplift: -0.0174pp.

Neither barrier passed the frozen signal rule.

## Decision

Retire this exact filing-semantic first-passage branch. Do not tune semantic
buckets, lookback windows, category thresholds, model capacity, barrier sizes,
feature subsets, or score quantiles on calibration outcomes.

The runner model shows a small and fairly broad ranking benefit, but it does not
produce an economically superior selected tail. Proceed to a different causal
state source.
