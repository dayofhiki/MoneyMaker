# Expanded-history past-only sequential Q v2.1 — pre-registration

## Status

Frozen before inspecting any September-December 2025 event, state, feature or
trading outcome.

This is the first model experiment under Historical Development Expansion v2.0.
It tests one primary intervention: substantially more chronological training and
calibration support. It must not access April 2026 or any later validation data.

## Fixed data set

Development months after successful v2.0 construction:

- 2025-09
- 2025-10
- 2025-11
- 2025-12
- 2026-01
- 2026-02
- 2026-03

The four 2025 months are adaptive development data, never future validation.

## Fixed information set

Preserve the v1.8 feature families and point-in-time semantics:

- existing causal state price/volume/breadth/sequence features;
- exact lagged FINRA short-volume family;
- exact strictly-prior 8-K family;
- exact publication-safe FINRA short-interest family.

The short-interest publication schedule is extended backward using documented
public dissemination dates before any 2025 state/model output is inspected.
Publication date must remain strictly earlier than the state trading day.

No NBBO feature is allowed in this branch because both configured historical
quote REST and quote Flat File paths returned HTTP 403 in v1.9.

No news/title text, ticker-details shares outstanding, feature subset search,
hand interaction search or new threshold is allowed.

## Model and action geometry

Reuse State selected-tail calibrated sequential Q v1.8 unchanged:

- checkpoints at 0, +5, +10 minutes from first eligible state;
- BUY holds exactly 10 minutes;
- WAIT advances exactly 5 minutes;
- SKIP value is zero;
- at most one BUY attempt per ticker-day;
- missing required checkpoint after WAIT terminates with SKIP;
- raw Q regressors, capacities, seeds and winsorization unchanged;
- chronological final 20% of training days is calibration partition D;
- remaining fit days are split into chronological A/B/C thirds;
- selected-tail correction uses calibration rows with raw predicted Q > 0;
- subtract max(0, daily mean optimism + 1.645 * SE);
- fewer than five selected calibration days disables that action;
- zero remains the action threshold.

No correction, threshold, capacity, timing, target or cost assumption may be
changed after results.

## Strict past-only evaluation

Evaluate only these three months:

### January 2026
Train/calibrate only on 2025-09 through 2025-12.

### February 2026
Train/calibrate only on 2025-09 through 2026-01.

### March 2026
Train/calibrate only on 2025-09 through 2026-02.

No future development month may enter any fit/calibration split. A provenance
table must prove max(training day) < min(evaluation day) for every fold.

Do not report September-December model results as if they were holdouts for this
experiment; they are training support.

## Comparators

For every evaluation month report:

1. earliest_eligible_10m_cap1;
2. raw_same_fit_cap1 using the exact same fit rows without selected-tail
   correction;
3. selected-tail calibrated sequential policy, the primary policy.

This isolates the effect of calibration from the effect of more history.

## Execution and accounting

Cost scenarios remain exactly the existing light/base/stress scenarios. No
post-result cost discount is allowed.

Persist every BUY attempt, including missing entry/reference labels.

Replay the primary policy through the frozen position-ledger accounting rules:

- independent synthetic account per evaluation month/scenario;
- existing $10,000 accounting unit and $1,000 fractional order unit;
- cash reservation, open-position block, delayed exit and unresolved-capital
  treatment unchanged;
- these units are audit conventions, not capital recommendations or claims of
  actual fills.

Bar-reference execution remains modeled, not observed NBBO execution.

## Required outputs

Report at minimum:

- fit/calibration/evaluation date provenance;
- selected calibration rows, days, optimism and corrections per action/fold;
- BUY attempts, evaluated attempts and every missing-evaluation reason;
- monthly trade count, gross/base/stress mean;
- day-balanced base mean, median, p05, severe-loss rate and worst day;
- selected predicted Q versus realized base;
- action path mix;
- external feature coverage;
- pooled trading-day-cluster bootstrap;
- frozen position-ledger summary for calibrated policy under all cost scenarios.

## Development promotion rule

Do not preregister or access a fresh validation month unless ALL of the
following hold for the calibrated primary policy:

1. at least 15 evaluated trades in each of January, February and March;
2. arithmetic base-net mean > 0 in every evaluation month;
3. day-balanced base-net mean > 0 in every evaluation month;
4. day-balanced base-net mean strictly exceeds earliest_eligible_10m_cap1 in
   every evaluation month;
5. base p05 and stress-net mean are not worse than earliest entry in any month;
6. pooled trading-day-cluster bootstrap 95% lower bound > 0;
7. short-volume latest-prior coverage >=90%, publication-safe short-interest
   latest coverage >=90%, and 8-K query completion =100% in every month;
8. frozen position-ledger BASE scenario for the calibrated policy has positive
   ending marked return in every month AND complete_accounting=true in every
   month.

A light-scenario profit cannot substitute for a failed base scenario. Zero
trades cannot pass. Missing references cannot be silently assigned zero P&L.

If any criterion fails, April 2026 remains sealed. Use the preregistered
diagnostics to decide whether the next intervention is state information,
execution protocol, candidate universe, or policy calibration. Do not tune this
exact branch on the failed results.



## Result

Workflow request 62 completed successfully after the evaluation loader was
changed to project Parquet inputs to model columns and retain only the causal
0/5/10-minute checkpoints. The failed requests 60 and 61 exited with code 143
while materializing full monthly panels. Request 61 proved that column
projection alone was insufficient; checkpoint compaction then made all three
monthly evaluations and the position-ledger replay complete. This isolates the
operational failure to runner resource pressure from accumulated full panels,
not an API, feature-join, date-leakage, or model exception.

Run: https://github.com/dayofhiki/MoneyMaker/actions/runs/35547029321

### Primary policy

| Month | Evaluated trades | Base mean | Day-balanced base mean | Account base marked return | Accounting |
|---|---:|---:|---:|---:|---|
| 2026-01 | 2 | -1.699% | -1.699% | -0.340% | complete |
| 2026-02 | 0 | n/a | n/a | 0.000% | complete, zero trades |
| 2026-03 | 0 | n/a | n/a | 0.000% | incomplete: one unresolved attempt |

The calibrated policy therefore failed the minimum-trade, positive-return,
bootstrap, and account-promotion criteria. Zero trades are not profitable
evidence.

### Raw same-fit diagnostic

| Month | Trades | Gross mean | Base mean | Day-balanced base mean | q0 Spearman |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 181 | -0.104% | -1.246% | -0.933% | 0.147 |
| 2026-02 | 87 | +0.271% | -0.877% | -0.462% | 0.221 |
| 2026-03 | 137 | +0.498% | -0.672% | -0.735% | 0.250 |

More chronological history improved broad holdout ordering but did not create a
stable cost-covering selected tail. Within raw BUY attempts, the highest
predicted-Q quintile remained base-negative in January (-0.984%) and February
(-1.446%); March was positive (+0.231%) but did not replicate across months.

Raw BUY-attempt evaluation was also incomplete: 58/239, 39/126, and 39/176
attempts were unevaluable in January-March, respectively. Missing entry
references and missing ten-minute exit labels must remain explicit.

### Calibration diagnosis

The selected-tail optimism corrections ranged from 1.651 to 3.480 percentage
points for BUY heads. Both WAIT heads had zero positive calibration
predictions in every fold, so the frozen rule assigned infinite correction and
disabled WAIT. The resulting policy was effectively conservative BUY-or-SKIP,
not a useful repeated BUY/WAIT controller.

### Data checks

External-data coverage passed: short-volume latest-prior and 8-K query
completion were 100% in every month; publication-safe short-interest coverage
was 98.7%-98.9%. Date provenance remained strictly past-only.

## Decision

Fail v2.1 development promotion. Keep April 2026 and later validation sealed.

Do not weaken the cost model, lower the zero action threshold, or reduce the
optimism correction on these results. More history alone did not repair the
unstable cost-covering tail. The next branch should test the previously observed
within-episode entry-rank signal with the expanded chronological history and the
already-fixed v0.3 gate. An explicit SELL/HOLD policy should only be evaluated
after a replicating cost-positive entry policy exists; otherwise exit freedom
would add degrees of freedom to a negative entry process.
