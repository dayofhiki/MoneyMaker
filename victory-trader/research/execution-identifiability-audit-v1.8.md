# Execution identifiability audit after v1.8

Date: 2026-09-19. Diagnostic only. No new model, threshold, trade selection,
cost discount, fresh month, data purchase or deployment was introduced.

## Source and reproduction
Completed request 47, run https://github.com/dayofhiki/MoneyMaker/actions/runs/35389089413
artifact 10565520659, SHA256
eb004ac0b980cbc4c047fd50e3d653f3a188e67dd70bd060549fae5d61464dfd.
Uses the six-decimal published text summary, not a recomputation of raw trades.

Run `python research/audits/execution_audit.py ARTIFACT.zip`.
Run tests from research/audits: `python -m unittest -v test_execution_audit.py`.
Three standalone tests passed locally: arithmetic, count mismatch rejection,
and incomplete-month rejection. These are separate from the prior 250-test CI.

## Findings: frozen raw-same-fit comparator
All returns are arithmetic means per evaluated trade, not portfolio returns.

| Month | Evaluated / attempted | Unevaluated | Gross mean | Base mean | Modeled friction | Descriptive reduction needed to break even |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jan | 291 / 348 | 16.38% | +0.273900% | -0.915769% | 1.189669 pp | 76.98% |
| Feb | 134 / 175 | 23.43% | +0.461146% | -0.743722% | 1.204868 pp | 61.73% |
| Mar | 152 / 192 | 20.83% | +0.476001% | -0.650326% | 1.126327 pp | 57.74% |

Modeled friction = gross mean minus base mean.
The descriptive reduction is 1 - gross / friction. It is an additive
sensitivity of recorded mean returns with trade selection fixed, NOT a claim
about feasible broker fees, spreads, parameter rescaling or executable profit.
Break-even average all-in cost budgets are 0.273900/0.461146/0.476001 pp.
The repository base scenario assumes each side has 25 bp half-spread and
25 bp adverse slippage, plus a one-cent minimum half-spread. These are explicit
scenarios, not observed fills.

Selected predicted net Q minus recorded net outcome is
+1.573016/+1.413842/+1.170192 pp. This overprediction exists relative to the
same cost-adjusted target used by the model; it cannot be explained away merely
by noting that the cost scenario may be conservative.

## Missing evaluation is not synonymous with no fill
The attempt evaluator rejects missing/nonpositive entry price OR missing
base/gross/stress return. Therefore the reported 138 unevaluated attempts
(57+41+40) cannot all be classified as unfilled orders.
Future label availability must never be used to choose a different ticker.

As a purely algebraic sensitivity, if every missing attempt had a comparable
realized net outcome, their means would need to be +4.675242/+2.430701/+2.471239%
to bring each month's all-attempt equal-weight mean to zero.
No such returns are observed or imputed. Some attempts may not be executable,
so that hypothetical denominator is not an estimate of portfolio P&L.

## Execution-information audit
Existing execution_nbbo_probe.py can use a quote up to one second AFTER the
target when no prior quote exists. This is labeled in its report, but those
fallback quotes must not be admitted as decision-time features.
Ask-to-bid quote returns also omit depth, queue position, order size and market
impact. A quote observation alone does not establish a realizable fill.

## Next implementation boundary
Before further model selection:
1. Split attempted-entry diagnostics into missing/invalid entry, missing exit
   label and missing scenario output, with mutually exclusive reason counts.
   Persist every attempt rather than only the evaluated trades.
2. Preserve the original universe and decisions while auditing those reasons.
3. A future execution-data probe must require prior timestamps, bounded quote
   age, valid uncrossed quotes and explicit size/depth limitations.
4. Check existing data entitlement evidence before acquiring quotes. Respect
   any authorization failure; do not change subscriptions or consume a new
   validation month without user direction.
5. Freeze cost-calibration and model-evaluation rules before a new experiment.

Status: this diagnostic audit is complete. No training run is active or scheduled
by this change. Profitability remains unsubstantiated. The next concrete task is
attempt-level missingness instrumentation, not another BUY/WAIT threshold search.
