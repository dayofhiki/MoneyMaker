# Requests 275–278 — causal pullback admission findings

## Executive conclusion

The current bottleneck is no longer basic exit timing or the existence of a
2% pullback/rebound trigger. The model can now reject a large fraction of bad
pullback states, but the surviving trades are still negative. The next
boundary is the pre-HOT candidate state and, specifically, the mismatch
between the candidate model's historical training target and the actual
downstream trading policy we now care about.

June15-19 remains unopened.

## Request275 — joint candidate/pullback admission

- Fixed teacher policy: first causal 2% pullback, -2.5% hard stop, 2% trailing
  drawdown, 10-minute exact max hold; missing deadline stays unresolved.
- May8 prediction diagnostics showed nontrivial ranking information:
  value Spearman 0.3904, positive AUC 0.5938.
- Admit-all May8 economics were poor: candidate mean -0.3288%, trade mean
  -1.7745%, trade positive rate 17.65%, severe-loss rate 54.41%.
- Admission filtering reduced candidate loss to roughly -0.006% to -0.013%
  across tested fractions, but no positive subset passed. Calibration failed,
  so May11-20 was not opened by Request275.

Interpretation: local pullback-state ranking exists, but the state/teacher
combination cannot yet identify positive trades.

## Request276 — causal pullback recovery confirmation

A pure price-action hypothesis was tested: after the 2% drawdown, wait for
0.5%, 1.0%, or 1.5% recovery from the running pullback low before entering.

Fit baseline:
- 73 entries
- candidate mean -0.2806%
- trade mean -1.1858%
- trade positive rate 16.98%
- severe-loss rate 47.17%

Recovery confirmation made results worse:
- 0.5% recovery: trade mean -2.0851%, severe 65.0%
- 1.0% recovery: trade mean -2.1577%, severe 67.65%
- 1.5% recovery: trade mean -2.7397%, severe 78.57%

No rule passed. May8 and May11-20 remained unopened by Request276.

Interpretation: simple rebound confirmation is not a recoverable-pullback
signal. It often enters a dead-cat bounce later and at a worse price.

## Request277 — strictly lagged completed-minute context

The timing contract was explicitly audited:
- raw broad-scan timestamp shift is exactly +60 seconds on 100% of rows;
- execution uses the open whose original bar_start_t equals decision_t;
- context uses raw scan.t == decision_t, therefore only the minute ending at
  decision_t, never the execution minute's future H/L/C/V.
- tests verified that changing the execution minute's future OHLCV does not
  change the decision input.

Leave-one-day-out May5-8 support:
- 1,602 candidate episodes
- 330 pullback states
- 259 resolved pullback outcomes

Baseline OOF representation:
- value Spearman 0.1583
- positive AUC 0.5005
- severe-loss AUC 0.5947

With completed-minute context:
- value Spearman 0.2502, gain +0.0919
- positive AUC 0.5317, gain +0.0312
- severe-loss AUC 0.6664, gain +0.0717

The representation gate failed only because absolute positive AUC remained
below the preregistered 0.60 floor.

Interpretation: completed-minute context materially improves return ranking
and, especially, the ability to identify dangerous pullbacks. It is much more
useful as a risk signal than as a direct buy signal.

## Request278 — risk-first admission

A small preregistered family combined the two Request277 strengths:
1. reject the riskiest 30% or 50% by predicted severe-loss probability;
2. among survivors, keep the top 30% or 50% by predicted return.

Raw OOF baseline:
- 330 admissions
- candidate mean -0.2666%
- trade mean -1.5761%
- trade positive rate 18.92%
- severe-loss rate 52.12%

Best candidate-mean combination was risk reject 50% / value retain 30%:
- 75 admissions
- resolution/cash 98.88%
- candidate mean -0.0351%
- day-balanced mean -0.0355%
- trade mean -0.9759%
- trade positive rate 24.56%
- severe-loss rate 38.60%
- candidate mean improvement vs raw +0.2315 percentage points

All four combinations substantially reduced harm, but every day remained
negative and every trade subset remained negative. The OOF economic gate
therefore failed and May11-20 was not opened by Request278.

## What this changes

Lower-level filtering is now demonstrably useful but insufficient. Continuing
to tune stop sizes, rebound percentages, or admission thresholds is unlikely
to solve the core problem.

The existing pre-HOT candidate model was trained against an earlier
fixed-first-watch target. That target is no longer aligned with the real task:
select a HOT candidate, wait causally, possibly enter, manage the position,
and end with positive economic value.

## Next boundary — Request279

Rebuild the candidate model against a frozen *downstream policy outcome*.

Proposed target:
- at HOT time, use pre-HOT information only;
- label the candidate by the realized economic result of one frozen causal
  downstream policy;
- no pullback / no admission is cash 0;
- admitted missing-deadline outcomes remain unresolved, never zero;
- no future-best entry or exit labels.

First perform leave-one-day-out May5-8 representation and economic-selection
diagnostics. Do not add new entry/exit knobs. If the candidate-level model can
concentrate positive downstream outcomes, then combine it with the already
validated Request277 risk context and only afterward restore recurrent
HOLD/EXIT.

Do not open June15-19 until this candidate-level boundary produces a stable
development policy.
