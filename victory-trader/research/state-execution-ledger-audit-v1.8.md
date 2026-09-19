# v1.8 execution ledger and chronological audit

Frozen 2026-09-19 before instrumented outcomes. This is an evaluation correction,
not a new profitable strategy claim. No new validation month or cost discount.

1. Preserve existing features, labels, seeds, Q gates and one-attempt semantics.
2. Record every BUY after its action was chosen. Retain decision time, stage,
   Q values, entry reference, all evaluation labels, overlapping missing flags,
   mutually exclusive reason and legacy evaluator inclusion.
3. Reason priority: missing entry, invalid entry, missing gross label, missing
   scenario label, nonfinite return, evaluated. Missing labels do not imply no fill.
   Nonfinite observations block promotion; instrumenting does not silently alter
   the historical evaluator or replace omitted returns with zero.
4. Run the unchanged LOMO development experiment and reconcile prior trade counts
   291/134/152 raw and 0/1/0 calibrated. Mismatch requires investigation.
5. Also run forward development: January-February only train/calibrate March.
   Refuse any training date >= first evaluation date. Report split provenance.
   March is already seen and identical to the March LOMO fold: this is a chronology
   guard, NOT an independent test, fresh holdout or extra evidence of alpha.
6. Persist complete attempts and reason summaries. No policy tuning on the output.

## Diagnosis and architecture gap
The current model's BUY correction is 2.51-7.56 pp based on sparse 8-9-day samples;
WAIT then loses positive continuation targets. This explains abstention, but
reducing the guard does not establish profitable EV. The raw comparator already
loses money. Daily one-attempt 0/5/10 checkpoints cannot be described as an
all-session continuous re-entry policy or portfolio-return backtest.
LOMO includes future months for January/February and is a development diagnostic.

## Path toward a continuously operating research system
First complete the attempt and chronology audit. Then design a separately frozen
past-only trained shadow policy with explicit position state, capital/concurrency,
re-entry rules, order expiry and session close handling. Maintain separate gross
alpha, observed execution-cost uncertainty and portfolio risk components. Require
causal liquidity inputs and authorized timestamped data, not future-label filters.
Assess cost-adjusted portfolio P&L, drawdown, turnover and coverage, not trade
count or prediction rank alone. A no-trade decision is legitimate where net EV
is not supported; minimum activity is an evaluation criterion, never forced BUY.
Live brokerage execution, new subscriptions and fresh validation data remain
outside this change. No claim of profitability is authorized by implementation.

## Monitoring
Hourly read-only market/research status watch is enabled in the conversation.
It reports material updates using available public observations and connected
GitHub state. It is not a streaming feed, deployed model, auto-retrainer or
order-execution service. No real-time fill or latency claim is made.

## Completed result — request 48
Run https://github.com/dayofhiki/MoneyMaker/actions/runs/35436142509 succeeded.
Artifact 10582810251:
sha256:23743477e822385dff3d494dbb7ed60de9a00857f0b99561bca2818d24da95dc.
CI run 35436085196 passed Ruff and 259 tests before execution.

| Raw policy month | Attempts | Evaluated | Missing entry | Valid entry, missing gross outcome |
| --- | ---: | ---: | ---: | ---: |
| January | 348 | 291 | 22 | 35 |
| February | 175 | 134 | 13 | 28 |
| March | 192 | 152 | 18 | 22 |
| Total | 715 | 577 | 53 | 85 |

All counts reconcile. The 85 gross-outcome-missing cases comprise 61.59% of
138 omitted evaluations, and 11.89% of all raw BUY attempts. They are NOT
confirmed unfilled orders. An entry bar exists but the fixed-horizon label is
missing. Actual fills remain unobserved. Their P&L direction cannot be inferred.

Raw evaluated trade counts and base means exactly reproduced request 47 at
reported precision: 291/-0.915769%, 134/-0.743722%, 152/-0.650326%.
Calibrated results likewise remain 0/1/0 trades; the February trade is -3.822639%.
There was no profitable-model improvement in this instrumentation run.

Forward provenance: fit 2026-01-02..2026-02-17, calibration
2026-02-18..2026-02-27, evaluation 2026-03-02..2026-03-31.
Forward March reproduces the same March LOMO fold, including 18 missing entries
and 22 missing outcomes. This is a chronology-path check, not fresh evidence.

## Concrete model-development priority after these observations
The data builder uses next-minute open for entry and exact horizon close for
the 10-minute label. A missing exact exit bar removes that outcome. Neither
the current count nor the previous report tells us whether the absence comes
from no trading, a halt, session boundary or a data-quality gap. Do not invent
a reason or impute zero P&L.

Before claiming continuous trading performance, implement a causal position
ledger and outcome-label audit that:
- records intended order, observed reference entry, pending exit, actual later
  observations and unresolved status separately;
- keeps an unresolved position and its committed capital outstanding instead
  of deleting it, resetting cash or calling it unfilled;
- distinguishes stale mark-to-market from realized sale, never retroactively
  sells at a last available bar after discovering the future is missing;
- uses observed information for eligibility, and future observations only for
  training labels and evaluation;
- blocks performance promotion whenever unaccounted entry/exit cases remain.

Then fit and evaluate the continuous-entry/exit policy against this ledger
using past-only data and unchanged cost scenarios. Separate estimated gross
return, cost/observability uncertainty and account risk. A model re-fit using
the current omitted-outcome sample alone is not an adequate fix.
No live trading, new quote subscription, new validation month, or streaming
scanner was enabled by this run. The hourly status watch remains read-only.
