# Request 209 — opportunity admission + transition timing

## Status

Pre-registered after Request 208.

Request 208 materially improved timing on the already-admitted population:
matched day-balanced BASE return improved by +0.1375pp, all 5/5 days improved,
and the day-cluster 95% interval was entirely positive. However the resulting
fixed-three-minute trades still averaged about -1.11% day-balanced.

The remaining question is therefore no longer only "when?". It is also
"should this watched episode be traded at all?"

No new dates are opened.

## Architecture

Two separate causal decisions are composed.

### 1. Minute-1 opportunity admission

At the first re-observed WAIT state, predict whether the episode contains
enough remaining executable opportunity to justify keeping it in the trading
pipeline.

Training target:

- inspect exact minute 1..5 candidate entry states;
- for each entry, compute the same fixed 3-minute BASE net return used by
  Requests 207/208;
- episode remaining opportunity = maximum valid return across those candidate
  entries.

The target is a diagnostic option-value label. It is not the realized policy
return.

Estimate expected opportunity using a hurdle model:

- calibrated probability that remaining opportunity > 0;
- expected magnitude conditional on positive;
- expected magnitude conditional on non-positive;
- expected opportunity = p * positive magnitude + (1-p) * non-positive
  magnitude.

Admission is semantic and frozen:

    ADMIT iff predicted expected remaining opportunity > 0
    otherwise ABSTAIN

No percentile threshold is tuned.

### 2. Entry timing

For admitted episodes, use the exact Request-208 transition representation and
runtime:

- minutes 1..4: ENTER if predicted ENTER-now minus WAIT-one-minute advantage
  >= 0, otherwise WAIT;
- minute 5: ENTER if evaluable.

## Evaluation

Comparator: Request-208-style transition timing without admission.

Primary realized metric: fixed-3-minute BASE net return of the executable
entry chosen by each policy.

## Frozen development gate

Pass only if:

- minute-1 admission-state coverage >= 80%;
- admission rate is between 5% and 70%;
- at least 30 executable trades;
- admitted policy day-balanced BASE mean > 0;
- admitted policy improves day-balanced BASE mean versus ungated transition
  timing;
- admitted policy positive on at least 4/5 days;
- day-cluster bootstrap 95% lower bound of admitted policy mean > 0;
- severe-loss rate no worse than ungated transition timing.

This remains development-only and promotion-ineligible.
