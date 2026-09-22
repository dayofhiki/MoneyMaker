# Persistent learned shortlist v1.1 — pre-registration

## Status

Frozen after request 127 showed that the rank-30 incumbent band preserved
capture but did not reduce observation demand enough, and before outcomes from
2026-03-18, 2026-03-19, 2026-03-20, 2026-03-23 or 2026-03-24 are inspected.
The request-127 dates are not reused for selection. April 2026 and later remain
sealed.

## Question

Does a structurally wider hysteresis band suppress minute-rank oscillation
enough for high-resolution observation while preserving runner coverage?

## Frozen policy

- Fit the unchanged market-wide hazard model only through 2026-01-16.
- Keep learned focus capacity 60 and shortlist capacity 20.
- Retain incumbents while their current focus-hazard rank is at most 40; fill
  open slots by current hazard rank and reset at each session boundary.
- Rank 40 is fixed as two times the observation budget, not selected by trying
  alternatives on request-127 or request-128 outcomes.
- Compare against stateless learned top-20 on the same fresh sessions.
- Fetch no one-second data. Unique shortlist ticker-days remain the proxy for
  high-resolution calls.

## Promotion gate

All request-127 conditions remain unchanged: at least 10 crossings per day,
pooled capture loss at most two percentage points, daily noninferiority on at
least four of five days, at least 95% capture conditional on focus, at least
25% fewer ticker-days, at least 70% mean retention, and occupancy at most 20.

If this gate fails, do not tune rank 40 on these dates. Replace rank-only
hysteresis with an explicit causal admission budget or residence-time policy on
a later untouched block.
