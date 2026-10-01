# R300 preregistration — causal clock-time and local price-context ablation

Status: implemented for one consolidated research dispatch. The complete local
execution reproduced R299 exactly, reconciled all 86 entries and completed all
nested folds. 35 tests and repository research lint passed. The local economic
gate failed; no feature, model, threshold or risk changes were made in response.
The following design was fixed after R299 failure and before R300 results.
Development only; official artifacts will provide the reproducible saved result.

## Question

Does explicitly representing elapsed-time momentum/activity and local price
structure improve reachable HOLD decisions beyond R299 D? Existing rolling
20-observation features often cover minutes rather than 20 seconds. This is a
testable representation hypothesis, not a claim that more features create edge.

## Frozen boundaries and dependencies

- Same 86 entries / 97 candidates, May5-8 only, frozen R296 entry gates.
- Same -2.5% BASE-net hard stop, costs, pending next-regular-open execution,
  caps, absorbing exits, reachable states, pi0 old10m/2% trail and two-stage
  nested policy-improvement targets as R299. No hard-stop widening or filters.
- Same model family, hyperparameters, episode/day weights, seed schedule and
  HOLD iff predicted advantage >0. No feature/threshold/horizon search.
- Reuse R299 run 36810713498 artifact plus R298 run 36801235949 raw seconds.
  R299 does not contain raw seconds: declare both dependencies explicitly.
- No market API requests or new dates. June HOLD and July-August stay sealed.

## Prespecified arms

A: reproduce R299 D's saved OOF predictions and replay exactly. Keep the original
20 features as the controlled baseline.

B: original 20 plus six clock-time features:

1. close return over 5 elapsed seconds;
2. close return over 20 elapsed seconds;
3. close return over 60 elapsed seconds;
4. count of completed observed seconds in the past 20 seconds;
5. count of completed observed seconds in the past 60 seconds;
6. volume-rate ratio: volume in latest 20s divided by volume in preceding 20s.

C PRIMARY: B plus two 60-second price-context features:

7. current close location in the min/max COMPLETED-CLOSE range over latest 60s;
8. current drawdown percentage from the maximum COMPLETED CLOSE over latest 60s.

Keep original event-count features rather than replacing them: A->B isolates
adding clock information, B->C isolates local context. Do not select whichever
arm wins. Eight added columns total; feature definitions frozen before dispatch.

## Exact causal feature contract

Let t be decision time at a completed second. Use raw regular-session bars with
bar start >= entry_t and bar end <= t only; exclude unfinished bars and never
use subsequent opens. A bar ending exactly at t is included. Reuse the causal
entry execution price as an anchor at entry_t.

- Return w: 100*(current close / latest known close at or before t-w - 1).
  Only valid if t-entry_t >= w; the anchor supplies the reference if no completed
  close exists before t-w since entry. No interpolation/future backfill.
- Latest-window membership: t-w < bar_end <= t. Empty time seconds mean no
  observed bar, not zero-return synthetic bars. Count observed bars, not n trades.
- Activity counts available from entry; existing position_age_s indicates the
  partial history. Window returns remain missing until the full elapsed horizon.
- Volume ratio compares (t-20,t] to (t-40,t-20], summing raw v. Require age >=40s
  and positive denominator; else NaN. Equal-duration windows make the volume
  sum ratio the rate ratio. Do not fill missing denominator using later volume.
- 60s range/drawdown require age >=60s. Range uses completed closes in (t-60,t],
  not highs/lows or future peaks. Location (c-min)/(max-min); flat range =0.5.
  Drawdown 100*(c/max-1). These are dimensionless and deterministic.
- Preserve genuine NaNs via the existing model preprocessing; do not drop any
  position or convert unavailable future information into a feature.
- Assemble features on the full causal position prefix, THEN retain reachable
  training states. Never reset history at a hypothetical voluntary exit.

## Learning and evaluation

For B and C independently execute the exact R299 inner/outer day exclusions.
Each arm's pi1 produces its own downstream labels, because its continuation
policy differs. Hold targets/risk/execution ALGORITHM fixed, not falsely claim
that realized pi1 labels are identical across arms. Use identical model seed
schedules across arms. Log all fits, feature availability and target provenance.

Report A/B/C mean, median, daily BASE, LIGHT/STRESS, CVaR5, positive rate, gross
and cost drag, exact all-entry reconciliation, hard-stop/model-exit counts,
hold durations, fill delays, largest-winner-excluded mean, risk-reachable upside
and original whole-path upside denominators separately. Report B-A, C-B, C-A
matched gains with both day and ticker-day cluster intervals. No selective
deletion of long fills/cost-only stops. Delay is realized outcome, not a feature.

For each arm report ordinary and day/episode-weighted OOF ranked advantage,
per-day weighted ranks, prediction/target sign distributions, and first-reachable
state versus all-reachable state results. Also report actually visited decision
states through that arm's submission as a distinct policy-conditioned diagnostic;
do not treat all counterfactual states as deployed decisions. No row bootstrap.
86 positions, 75 reachable trajectories and four days remain the real sample.

Primary C must meet ALL preregistered gates:

- All 86 entries reconciled, >=95% resolution; exact A replay within 1e-9.
- Positive BASE mean; positive BASE on >=3 of the 4 outer days.
- Positive paired gain vs frozen old trailing and vs R299 D/A.
- Positive BASE mean with largest winner removed.
- Positive day/episode-weighted OOF ranked continuation advantage.

Confidence intervals are uncertainty diagnostics, not proof from four days.
Passing only permits freezing for additional May development evaluation; it does
not certify profitability. B is an explanatory ablation, never a fallback winner.

## Required tests before execution

- Exactly 20s vs 20 observed bars differ on a sparse synthetic path.
- Window-end ties, entry anchor, incomplete history, no-price-observation gaps,
  zero volume denominators and flat close range behave as defined.
- Future bar/volume mutations leave all earlier feature values and actions
  unchanged. Label/fill mutation may change later outcomes only.
- No unfinished/outside-session bars; no group/day/position boundary leakage.
- Stop/cap precedence and absorbing pending behavior remain R299-compatible.
- Nested fit provenance excludes target/test days; all 86 retained, including
  eleven positions with no first reachable HOLD state and nine cost-only stops.
- Reproduce A return/submission/reason exactly; strict JSON and deterministic
  feature availability counts. No post-result repair disguised as tuning.

## Decision after result

If C fails, keep June sealed. Distinguish clock representation failure from
target-policy instability, liquidity gaps and entry/cost economics; propose a
separate preregistered diagnostic rather than expanding features indefinitely.
If C passes, freeze implementation and schedule a separately authorized later-May
development replay; those dates are previously explored, not pristine holdout.
Opening June requires a later explicit full-policy/evaluation freeze.
