# Persistent learned shortlist v1.2 — pre-registration

## Status

Frozen after request 128 passed capture and retention gates but failed the
observation-demand gate, and before outcomes from 2026-03-25, 2026-03-26,
2026-03-27, 2026-03-30 or 2026-03-31 are inspected. Earlier March blocks are
not reused for selection. April 2026 and later remain sealed.

## Question

Can an explicit causal admission budget meet the high-resolution resource
constraint without materially degrading runner coverage?

## Frozen policy

- Fit the unchanged market-wide hazard only through 2026-01-16.
- Keep focus capacity 60, shortlist capacity 20 and the already-tested rank-40
  incumbent band.
- Admit at most 300 distinct high-resolution symbols per session, equal to 15
  complete rotations of the 20-slot observation budget.
- After the budget is exhausted, fill from already admitted current-focus
  symbols. If fewer than 20 are present, keep the best causal incumbent rows
  with their last known score until an admitted symbol can replace them.
- Reset the admission ledger and incumbents at every session boundary.
- Fetch no one-second data; unique ticker-days remain the high-resolution-call
  proxy.

The 300-symbol ceiling is a resource policy fixed before evaluation, not a
threshold selected from request-129 outcomes.

## Promotion gate

Keep every prior condition unchanged: at least 10 crossings per day, pooled
capture loss at most two percentage points, daily noninferiority on at least
four of five days, at least 95% capture conditional on focus, at least 25%
fewer ticker-days, at least 70% mean retention, and occupancy at most 20.

If the gate passes, integrate this policy with one-second HOT allocation on a
new preregistered period. If it fails, do not alter the 300-symbol budget on
these dates; attribute the failure to capture or resource feasibility before
moving forward.

## Result — request 129

The first-come budget failed. It capped demand at exactly 1,500 ticker-days
versus 2,796 for stateless top-20, a 46.35% reduction, and retained 74.45% of
the prior set. But exact-prior capture collapsed from 53.56% to 40.17%, was
noninferior on zero days, and retained only 68.97% of learned-focus captures.

The fixed budget was consumed too early and prevented late-session admissions.
The resource ceiling remains viable, but first-come allocation is rejected.
