# Recurrent trader implementation audit v1.0

## Scope

This audit was performed after request 144 failed to produce an artifact. It
reviews the active first-HOT -> POSITION observability path for performance,
causality, chronological integrity, execution-label consistency, and whether
performance optimizations can change model semantics.

No new market date is opened by this audit. May 21 and later remain sealed.

## Findings and fixes

### A. One-second search optimization is semantics-preserving

The position path only consumes completed one-second aggregates satisfying:

    t >= decision_t - 60 seconds
    t + 1 second <= decision_t

The old implementation filtered the full ticker-day second frame with that
boolean predicate on every decision row. Searchsorted now finds the exact same
inclusive slice before running the unchanged feature calculations.

A regression test compares the optimized slice with the original boolean-mask
boundaries, including both endpoints and exclusion of the second beginning at
the decision timestamp.

This optimization changes lookup complexity only.

### B. Cross-session lag contamination existed

`_annotate_scan` historically grouped lagged minute features only by ticker.
That was harmless when callers passed one day at a time, but request-144-style
multi-day concatenation could make the first bar of a new session use the prior
session's final minute as its "previous one minute" observation.

This is not future leakage, but it changes the semantic meaning of:

- minute_return_1m_pct;
- return_accel_1m_pct;
- volume_ratio_prev1;
- transactions_ratio_prev1;
- already-runner state.

The shared annotator now groups by `trading_day, ticker` whenever a trading-day
column is present. Single-day historical behavior is unchanged. A regression
test verifies that lagged features and runner state reset at a session boundary.

### C. A genuine future execution-price leak existed in the position feature set

At position decision timestamp `state_t`, the completed minute ending at
`state_t` is causal information. The open of the minute beginning at
`state_t` is the next executable reference after that decision.

The old draft used that next open both:

1. as the hypothetical EXIT execution reference; and
2. as model input through `log_current_open`,
   `entry_to_current_open_pct`, and peak/trough calculations using that open.

That lets the model inspect its own next execution price before deciding
HOLD/EXIT.

The corrected causal feature set removes all next-open inputs. It uses the
completed bar close for current-price features and for drawdown/recovery.
The next open remains available only to labels and execution accounting.

Tests explicitly forbid execution/future-return columns from MODEL_FEATURES.

Request 144 produced no result artifact and is not evidence. Its interrupted
runs must not be revived as a valid evaluation.

### D. The historical request-140B selection threshold depended on future label availability

Request 140B trained its classifier on economically labeled fit episodes, which
is necessary for that diagnostic. But its diagnostic top-quartile threshold was
also calculated after dropping calibration rows whose future economic label was
missing.

For an executable policy this is future-dependent selection: at decision time
the trader cannot know whether a later oracle label will exist.

The historical request-140B artifact is not rewritten or retroactively
promoted. A new policy-safe selector keeps the same causal classifier training
but freezes its probability quantile over **all calibration feature rows**,
regardless of future label availability.

A regression test constructs missing future labels in calibration and verifies
that the threshold still uses all calibration rows.

### E. Request 144 materialized far more market state than the position model needed

The old path:

1. built 15 full market sessions;
2. concatenated them into one large DataFrame;
3. annotated the whole concatenation;
4. converted essentially every market row into lookup dictionaries;
5. built ticker-group DataFrames for the entire market.

Only first-HOT ticker-days can ever become position paths.

The corrected path streams one trading session at a time. Cross-sectional
attention rank is still computed using the complete market for that timestamp,
then only anchored ticker-days are materialized into position lookup state.
The full day is released before the next session is processed.

This preserves all cross-sectional information while greatly reducing peak
memory and duplicated objects.

The audit now reports full-market rows versus materialized position-market rows.

### F. Repeated position calculations were reduced without changing targets

For each position episode the old code repeatedly:

- rescanned completed minute history to recompute running high/low;
- rescanned all future opens to recompute the best remaining exit;
- recreated timezone conversions for the same market timestamp.

The corrected implementation:

- updates running high/low incrementally as completed bars arrive;
- precomputes BASE returns for available future opens and a suffix maximum;
- uses binary search for the strictly-later remaining-option label;
- caches session-clock conversion by timestamp.

The definitions of HOLD advantage and remaining option value are unchanged,
apart from removal of the invalid next-open feature exposure described above.

### G. Execution-cost labels are locked to the shared model

A regression test verifies that the position helper produces the same BASE
round-trip return as `net_round_trip_return_pct` for the same entry and exit
reference prices.

### H. GitHub reruns preserve the original head SHA

Request 144 attempt 2 was a rerun of workflow run 35859765163. GitHub reran the
original request head SHA rather than the later optimization commit. Therefore
that attempt did **not** contain the subsequent second-window optimization.

A corrected diagnostic must be launched as a new request/run from current main,
not as another rerun of request 144.

## Research consequence

The next no-new-date position observability diagnostic must use the corrected
causal implementation and the policy-safe entry selector. It may reuse only
the already-opened Apr30-May20 sessions.

May 21 and later remain sealed.
