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

7. Request-196's `strong_winner_3` label means that a +3% BASE opportunity
   exists at some later executable point. It does not require that the +3%
   opportunity occurs before a -3% risk stop would have fired. Therefore the
   Request-196 winner score is an observability signal, not yet a
   risk-compatible ENTER target. The next executable entry design must use
   path-ordered/barrier-aware economics (for example, reward reached before
   stop, or a full causal policy replay) rather than treating every eventual
   +3% oracle winner as capturable.

8. Legacy Request-189/192 trajectory helpers contain the same historical
   "missing state -> fallback exit" completion convention. Those helpers may
   remain useful for reproducing their original research reports, but they must
   not be reused as the execution engine for the next MoneyMaker policy.
   New executable policy code must distinguish:
   - no causal state observed: no new trading decision;
   - EXIT decision observed but execution unavailable: pending/unresolved
     execution;
   - exact research-cap execution: resolve only from an exact causal execution
     reference, not from an earlier data gap.

## Goal

The next executable experiment should be judged only by causal, realizable
account economics after modeled costs. Oracle opportunity, fresh diagnostic
rankings, and calibration policy selection must remain clearly separated.


## Additional structural findings from the final Request-194 audit

### 4. The current stop/take overlay is minute-close risk management, not a true hard stop

Request 194 evaluates the mark only on completed minute POSITION states. A stock
can cross a stop or profit barrier intraminute and reverse before the minute
closes without triggering the rule.

Therefore labels such as "hard_stop" currently mean "minute-close stop rule".
They must not be interpreted as broker-style stop-market behavior.

Before promotion, the risk engine should replay barriers on the available
one-second path (or finer live data) and execute only after the causal trigger
time.

### 5. Synthetic execution-cost predictability is not real spread/impact predictability

The BASE execution scenario is generated from fixed half-spread/slippage
parameters plus a minimum-cent spread floor. Consequently a high correlation
when predicting synthetic drag can largely reflect deterministic price-level
effects.

Do not claim that real market execution cost is solved from the Request-184/195
drag Spearman results. Final validation needs latency and friction sensitivity,
and live/paper execution should later replace synthetic spread assumptions.

### 6. The next entry target must be risk-compatible and path ordered

The next model should predict an economically attainable outcome under the same
risk rules it will actually execute. A useful target is not merely "a +3% exit
exists later"; it must encode whether reward is reachable before the stop or
otherwise replay the full frozen causal policy.

### 7. One ticker must correspond to at most one live position unless scaling-in is explicit

The attention runtime has a single POSITION state per ticker. The reference
account previously allowed overlapping episodes with different HOT timestamps
for the same ticker. The ledger now blocks overlapping same-ticker admissions.

Scaling in can be researched later, but it must be a deliberate action, not an
artifact of replay episode construction.

### 8. Reference-account equity is not yet a production-grade account model

The Request-194 account is intentionally interpretive. It sizes from cash plus
remaining cost-basis principal and reports realized-ledger drawdown rather than
continuous mark-to-market equity. It also assumes fractional sizing and ignores
FX presentation effects.

Before an untouched-date profitability claim, use one canonical account engine
with:
- mark-to-market equity and drawdown;
- one-position-per-ticker invariant;
- explicit whole-share/fractional-share assumptions;
- explicit USD accounting, with KRW only as a display conversion if needed;
- risk-normalized sizing rather than fixed 20% allocation alone.

The existing `position_ledger.py` already has better mark-to-market mechanics
and should be the base for consolidation rather than maintaining multiple
independent ledgers.

### 9. Stop-distance comparisons should be risk-normalized

A fixed 20% allocation with a -3% stop risks much less account capital than the
same 20% allocation with a -7% stop. Comparing those policies without adjusting
position size mixes exit quality with different risk budgets.

Future risk-policy comparisons should freeze an account-risk budget per trade
and derive position size from the stop distance, subject to liquidity and
capital caps.

### 10. Repeated development rows are correlated

Request-196 reports state-level AUC over many shadow states from the same
episodes. Those rows are useful for recurrent observability research but are not
independent trades.

Executable evaluation must:
- allow only the causal actions the runtime could actually take;
- avoid counting multiple high-score states from one episode as independent
  successes;
- cluster uncertainty at least by trading day and preferably by ticker episode
  where appropriate.

### 11. The broad research session is currently regular-hours only

The market-wide replay is intentionally restricted to the regular U.S. session.
This is not leakage or a code bug, but it is a scope gap for a surge-stock trader
because premarket can contain important discovery and price-formation
information.

Premarket should be a separately validated extension after the regular-session
policy is structurally sound, not silently mixed into the existing history.

### 12. Use one canonical execution/replay contract going forward

Multiple historical research modules implement their own missing-state,
delayed-fill, cap, and ledger rules. This flexibility helped exploration but
also created semantic drift.

The next executable policy should use one shared execution contract for:
- decision timestamp;
- first legally usable execution timestamp;
- delayed/unfilled orders;
- partial fills/exits;
- stop/take/trailing triggers;
- position uniqueness;
- account cash/equity.

Legacy modules can remain reproducible research artifacts, but they should not
be imported as the live-policy execution engine.
