# R299 design — reachable-state and risk-consistent HOLD action values

Status: designed, not dispatched. One consolidated research run when authorized.
Reuse R298 raw seconds and R298/R297 states; no repeated API acquisition required.
May5-8 only for the first ablation. June HOLD and July-August remain sealed.

## Hypothesis

R298 restored execution coverage without restoring HOLD predictability. 71.61%
of state rows are after a mandatory stop; horizon labels can value forbidden
post-stop rebounds. Separate these errors before adding features or model size.

## Frozen boundary

Same 86 entries/97 episode denominator; R296 setup/entry gates; disabled override;
20 causal HOLD features; gradient-boosting family, hyperparameters and seeds;
BASE/LIGHT/STRESS costs; -2.5% modeled-net hard stop; corrected R298 pending orders,
clock timers and regular-session fill references. No entry filters, stop widening,
threshold grids, best-peak target, new market dates or fresh-data tuning.

## Primary ablation arms

A. Reproduce R298 refit dynamic 300s exactly from the saved artifact.
B. Same 300s target/model, but train only reachable HOLD states: completed states
   strictly BEFORE the first mandatory stop; mandatory-stop states are forced
   EXIT and not learned HOLD examples. States after that submission are absorbing
   EXIT_PENDING/FILLED states, never resurrected positions. Keep all 86 positions
   in the outcome ledger, including cost-only stopped entries.
C. B plus risk-respecting 300s target: if a mandatory stop occurs before the
   target liquidation timer, its original stop submission and pending fill are
   the future liquidation value. Otherwise use the exact future timer and pending
   fill. Same interpretation applies to the 60s sensitivity, reported without
   choosing the better horizon. This isolates label/risk alignment from row removal.
D. Primary new policy: risk-respecting policy-improvement HOLD value. Decide
   whether observing the next active second and then following a downstream policy
   is better than EXIT now, rather than forecasting liquidation at a fixed minute.
   Details below. B/C are diagnostic ablations, not a result-based strategy search.

## D. Two-stage fitted policy improvement, with nested day exclusion

Use bounded two-stage improvement, not a claim of converged optimal control.

1. Reference policy pi0 is the frozen old 10m/2% trailing policy, with the same
   common hard stop and corrected pending fills. For every reachable state s,
   its WAIT label is downstream pi0 liquidation after the next completed active
   second minus liquidation from submitting EXIT now. pi0 uses only observed
   prefix information; future realized fills are training labels only.
2. For each outer held-out development day, leave that day out of ALL training
   and target-policy fits. Within the other three days, leave one inner day out,
   fit pi1 on the other TWO days using pi0-based labels, then replay pi1 on that
   inner day to produce WAIT labels for its reachable states. pi1 re-evaluates
   each observed second; only the research/session cap is a timeout.
3. Concatenate the three inner-day targets and fit pi2 on the outer training
   days. Test pi2 on the untouched outer day. A pi1 target policy never trains on
   the day on which its downstream outcomes are used as pi2 labels.
4. pi2 is the primary final policy for this experiment. It can differ from pi1;
   these are two bounded policy-improvement stages, not a fixed-point proof.
   Report that limitation. No additional iterations or threshold selection after
   looking at outer results. HOLD iff predicted advantage >0; risk forces EXIT.

## State and transition contract

- Reachability is defined by the first causal mandatory stop, independent of
  the learned voluntary exit. Voluntary post-exit states may be used as training
  counterfactuals ONLY while the underlying position remains risk-reachable.
- A next state is the next completed OBSERVED second. Elapsed wall-clock time
  and event gaps retain their real values; no fictitious price updates in silence.
- If a clock cap precedes the next observed state, WAIT submits liquidation at
  that cap. If the next state first triggers risk, it submits a forced stop then.
- Once EXIT is submitted, pending execution is absorbing; pay the following raw
  regular-session open, full gap and friction. Do not let future recovery cancel
  the pending order or become a HOLD target.
- At the cap/session end, no continuation action is available. Unknown final
  liquidation remains explicitly censored; do not fabricate endpoint prices.
- Trailing reference-policy peaks use the original position prefix, not a peak
  reset at a hypothetical WAIT start. All future outcomes are labels only.
- Do not use retrospective best-before-stop values as supervised targets; retain
  them only as a diagnostic ceiling with explicit future-knowledge attribution.

## Evaluation

Compare A/B/C/D on identical 86 entries using corrected execution. Also show
immediate, frozen old trailing, fixed300s and terminal-plus-stop benchmarks.
Report outer-day sample sizes, reachable/forced/post-stop states, target support,
weighted OOF signals, actions and pending delays, BASE/LIGHT/STRESS returns,
daily means, median, CVaR5, gross/net upside capture, left-on-table observability,
matched gains with day/ticker-day bootstrap, trade concentration and
leave-largest-out returns. Separate effects B-A, C-B and D-C.

Primary preregistered gate for D:
- >=95% resolution with all 86 positions reconciled;
- positive BASE mean and positive BASE on >=3/4 held-out days;
- positive paired gain versus old trailing and R298 A;
- positive BASE mean after removing the largest winning position;
- positive ranked WAIT/EXIT advantage on eligible outer states.

Four reused days and non-nested upstream entry OOF remain exploratory. Passing
this gate is permission to freeze D for further development evaluation, not a
profitability certification or a reason to choose another arm from its result.

## Required tests

- First stop ends reachability; no later rebound contributes to HOLD labels.
- Unchanged-price cost-only stops remain forced and counted, not removed.
- Future stop gaps are paid at pending next opens; post-submit policy cannot hold.
- Cap timer before next observed second takes precedence over later data.
- pi0 trailing peak retains the full prior position history.
- Inner/outer day provenance prevents target-policy training on labeled/test days.
- Future mutations change labels/fills but not earlier features/actions.
- Reproduce R298 arm A within 1e-9, preserve all entries, no outside-session fills,
  and serialize summaries with strict JSON.

## Next boundary

If D fails, analyze whether reachable-state signal, policy targets, price context,
or candidate economics is limiting; do not open June or widen the stop based on
this result. If D passes, freeze the full implementation before a separate
May11-20 development replay with May5-8-trained entry and HOLD models. Those later
May dates were previously explored elsewhere, so they are additional development,
not a pristine untouched holdout. Use the frozen R163 upstream contract and
verify data availability/provenance before that subsequent run. June HOLD is
scored only after the final policy and complete evaluation contract are frozen.
