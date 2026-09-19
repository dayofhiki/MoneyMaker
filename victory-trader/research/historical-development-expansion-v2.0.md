# Historical development expansion v2.0 — pre-registration

## Status

Frozen before acquiring or inspecting the September-December 2025 event outcomes.

This is a development-data expansion, not a trading-model result and not a
fresh validation. January-March 2026 remain adaptive development/evaluation
months already seen many times. No April 2026 or later month may be accessed by
this protocol.

## Motivation

The v1.8 calibration audit had only 8-9 selected calibration days per fold and
showed selected-tail optimism of roughly +0.87 to +4.06 percentage points.
Current quote entitlements cannot provide historical NBBO: REST and quote Flat
File probes both returned HTTP 403. The next identifiable intervention is more
chronological development support, not another gate/label rewrite or an
unobserved execution-cost discount.

## New development period

Acquire the complete full-universe event data for:

- 2025-09-01 through 2025-12-31 inclusive.

Reasons fixed in advance:

- four additional calendar months roughly double the number of available
  chronological development trading days relative to January-March alone;
- the period is immediately prior to the existing 2026 development months,
  limiting unnecessary long-horizon regime drift;
- September 1 is a calendar boundary chosen before seeing any event or strategy
  result from the new period.

All September-December 2025 data become development data immediately upon
access and can never later be described as unseen validation.

## Acquisition rules

Use the existing `victory-trader-research.yml` full-universe Flat File builder
unchanged:

- price universe: existing $0.50-$20 defaults;
- minimum intraday high return: existing +10% default;
- common-stock / split-day rules unchanged;
- no `max_candidates` debug sampling;
- 35-day historical context unchanged;
- unadjusted source prices and existing event-study construction;
- halt annotation may be attempted exactly as existing workflow specifies;
- chunking is an execution detail only and must not alter rows.

The aggregate output must pass the existing dataset audit. If acquisition is
partial or a chunk fails, do not silently continue with the surviving subset.

## State-panel preparation

After successful event acquisition, use the existing `state_panel.py`
construction unchanged. The generic Flat File event builder has the state-panel
keys `trading_day`, `ticker`, `timestamp_ms`, `threshold_pct`, and
`previous_close`; no outcome-dependent row filter may be added.

Split the resulting state panel by calendar month:
2025-09, 2025-10, 2025-11, 2025-12.

Then add the same causal breadth and sequence enrichments used by v1.8. Static
external families may be added only after their point-in-time availability is
verified for the newly covered dates.

## Future evaluation protocol

No strategy is selected in this acquisition step.

The first model experiment after data construction must be separately
pre-registered and strictly past-only:

- January 2026 evaluation: train/calibrate on Sep-Dec 2025 only;
- February 2026 evaluation: train/calibrate on Sep 2025-Jan 2026;
- March 2026 evaluation: train/calibrate on Sep 2025-Feb 2026.

No leave-one-month-out future training is allowed.

The initial comparator should preserve the frozen v1.8 raw/calibrated
sequential-Q geometry and cost scenarios so that added history is the primary
intervention. Any feature-set change requires its own preregistration.

## Accounting boundary

All future policy outputs must be replayed through the explicit position ledger.
Attempted orders, missing entry references, delayed exits, unresolved positions,
capital locks, turnover and drawdown must remain visible. Bar-reference fills
remain modeled observations, not claims of brokerage realizability.

A no-trade policy is not evidence of alpha. A cost scenario cannot be reduced
post hoc to manufacture profitability.

## Validation seal

No April 2026 or later data may be acquired for model validation until a
separately frozen past-only development branch passes its pre-registered
economic and accounting criteria. At that point exactly one validation month
must be registered before access.

