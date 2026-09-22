# Paced shortlist admission budget v1.3 — development protocol

## Status

This is a mechanism-development check, not a promotion test. Request 129 showed
that a 300-symbol ceiling meets the resource target but first-come allocation
creates late-session blindness. April 2026 and later remain sealed.

## Development split

- Fit the unchanged market-wide hazard on 2026-01-02 through 2026-01-09.
- Evaluate the mechanism on 2026-01-12 through 2026-01-16.
- These dates have appeared in earlier research and cannot establish final
  generalization. They are used only to reject or retain the mechanism before
  spending the sealed holdout.

## Frozen mechanism

- Keep focus 60, shortlist 20, rank-40 incumbent hysteresis and the 300-symbol
  session ceiling.
- Unlock 20 symbols at the first decision, then release the remaining 280
  admission slots linearly across the session, reaching 300 only at the final
  decision.
- When no unseen admission is unlocked, use already admitted current-focus
  symbols, then causal stale incumbents to maintain 20 observation slots.
- Reset all state at the session boundary.

## Development viability criteria

Use the same numerical conditions as requests 127–129: capture loss at most two
percentage points, daily noninferiority on four of five days, at least 95%
conditional focus capture, at least 25% fewer ticker-days, at least 70%
retention, and occupancy at most 20. Passing only authorizes a separately
preregistered sealed evaluation; it does not promote the model.

## Result — request 130

The paced budget failed the development viability criteria, so the sealed
holdout remains unopened. It reduced unique ticker-days from 2,518 to 1,467
(41.74%) and raised retention from 58.26% to 80.63%, but exact-prior capture
fell from 61.57% to 52.60%, an 8.97 percentage-point loss. Conditional capture
versus learned focus was only 82.47%, and no day met the two-point
noninferiority rule.

Together with request 129, this shows that the 300-symbol ceiling itself—not
only first-come timing—is incompatible with the current cross-sectional signal.
The admission-cap family is rejected without spending April data.
