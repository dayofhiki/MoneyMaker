# Integrity audit after Request 196

This audit was performed before opening any new dates or implementing the next
executable entry policy.

## Corrections made

### 1. Silent/missing minute states are not liquidation signals

Request 194 previously converted a missing next POSITION minute into an
automatic exit at the next available executable reference. That was a data
completion convenience, not part of the intended trader policy.

The corrected risk overlay now:
- keeps the position through silent/missing bar states;
- resumes decisions only when a later causal state is actually observed;
- does not manufacture an EXIT merely because a minute row is absent;
- leaves the trajectory unresolved if the exact forced-cap execution cannot be
  established honestly.

A missing observation can be modeled later as an explicit liquidity/state
feature, but it must never silently become a trading action.

### 2. Repricing lags use elapsed time, not row count

Request 195 used row shifts for nominal 1m/3m/5m repricing features. Sparse
microcap paths can omit silent minute rows, so "shift(3)" is not necessarily
"three minutes ago".

The corrected implementation requires an exact matching episode/state timestamp
at t-1m, t-3m, and t-5m. If that exact causal state does not exist, the lag is
missing rather than backfilled from an older row.

Acceleration likewise compares exact adjacent one-minute moves.

### 3. Sequence lag construction is vectorized

Request 196 repeatedly inserted one lag column at a time, fragmenting the pandas
DataFrame and producing performance warnings. The corrected implementation
builds each lag block in one reindex operation and concatenates blocks once.
Semantics remain exact-time and causal.

## Methodological guards for the next request

1. Request-178 June 23/24/25/26/29 remains development-only. It must not be used
   as promotion evidence.

2. Request-196's fresh top-quartile cutoff is diagnostic only. No executable
   ENTER threshold may be copied from that fresh quantile.

3. Request 196 trained its observability classifiers on the combined historical
   fit + calibration blocks. Therefore those exact fitted models must **not** be
   reused while also selecting an action threshold on the same calibration
   block.

4. Any executable Request-197 style policy must retrain the entry model on the
   historical fit block only, then use the untouched chronological calibration
   block solely to freeze ENTER / WAIT / ABSTAIN thresholds and other policy
   choices.

5. The Request-194 partial-take/trailing policy did not establish positive
   economics. Its exact -3/+10/5 specification is not a validated profit rule.
   Only the broader principle that explicit downside control reduced severe-tail
   loss is considered reusable evidence. Any exact risk rule used by a future
   executable policy must remain calibration-frozen and be re-evaluated as part
   of that policy.

6. Missing market activity must be represented as information, not implicitly
   converted into HOLD/EXIT labels or actions.

## Goal

The next executable experiment should be judged only by causal, realizable
account economics after modeled costs. Oracle opportunity, fresh diagnostic
rankings, and calibration policy selection must remain clearly separated.
