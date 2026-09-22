# Persistent learned shortlist v1.0 — pre-registration

## Status

Frozen after request 126 passed its performance gate but exposed excessive
observation turnover, and before outcomes from 2026-03-11, 2026-03-12,
2026-03-13, 2026-03-16 or 2026-03-17 are inspected. Earlier evaluation dates
are not reused for tuning. April 2026 and later remain sealed.

## Question

Can a causal incumbent buffer reduce learned top-20 ticker-day demand and set
turnover while preserving exact-prior-minute runner coverage?

## Frozen policy

- Fit the unchanged market-wide hazard model only through 2026-01-16.
- Keep learned focus capacity 60 and shortlist capacity 20.
- At each minute, retain prior shortlist incumbents while their current learned
  focus-hazard rank is at most 30; fill remaining slots by current hazard rank.
- Reset incumbents at every session boundary.
- Compare against the stateless learned top-20 from request 125/126 on the same
  fresh sessions.
- Fetch no one-second data in this isolated bridge. Unique shortlist
  ticker-days are the preregistered proxy for required high-resolution calls.

The rank-30 buffer is fixed before evaluation. It is a 10-name hysteresis band
around the 20-name observation budget, not a value selected from the new dates.

## Promotion gate

Promote persistent top-20 selection into the full hierarchy only if all are
true:

1. every session has at least 10 first +10% crossings;
2. pooled persistent exact-prior-minute capture is no more than 2 percentage
   points below stateless top-20;
3. the same 2-point noninferiority condition holds on at least four of five
   sessions;
4. persistent capture conditional on focus is at least 95%;
5. unique ticker-day demand falls by at least 25%;
6. mean minute-to-minute shortlist retention is at least 70%;
7. shortlist occupancy never exceeds 20.

If the gate fails, do not tune the rank buffer on these dates. Attribute the
failure to capture loss or inadequate demand reduction and test a different
causal persistence mechanism on a later untouched March block.
