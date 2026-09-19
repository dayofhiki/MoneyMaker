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
