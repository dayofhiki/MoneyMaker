# Request 193 — pre-entry shadow-path delayed viability observability

## Status

Pre-registered after Request 192 failed its frozen executable development gate
and before implementing or running Request 193.

Request 191 showed that future cost-cover viability becomes strongly observable
after causal path information arrives. Request 192 showed that using that signal
only after paying the original entry cost is too late: the composed controller
remained economically negative and was slightly worse than minute-1 liquidation.

Request 193 therefore moves the same idea upstream. A HOT/BUY candidate is
followed as a **shadow position without buying it yet** for minutes 1-10. At
each completed shadow state, ask whether entering at the next executable open
would still leave a cost-covering exit before the inherited 30-minute research
cap.

No new dates are opened and no executable entry policy is changed in this
request.

## Population

Historical fit/calibration:
- Request-171 BUY-gated rich-second POSITION rows;
- Request-163 historical minute scan for exact executable open references;
- Request-173 market-regime context.

Fresh development evaluation:
- Request-178 POSITION + scan shards on Jun 23/24/25/26/29.

Use only causal shadow states with minutes_held in [1, 10].

## Shadow semantics

The existing POSITION path columns are reused only as causal price-path
descriptors. The original entry open is treated as an observable HOT-time price
anchor, not as a paid transaction. No original-entry P&L or future execution
reference is an input feature.

For a shadow state at time t:
- delayed entry is the next executable open after the completed state;
- candidate exits are later executable opens strictly after delayed entry;
- the final candidate exit is bounded by the original HOT + 30-minute research
  cap.

All delayed entry/exit prices are labels/outcomes only.

## Targets

For each state:

    delayed_future_best_base_return_pct =
        max BASE return from delayed next-open entry
        to any later executable open through HOT+30m

    delayed_future_cost_coverable =
        1[delayed_future_best_base_return_pct > 0]

This recomputes modeled BASE costs from the delayed entry price. It does not
reuse Request-191's original-entry target.

## Causal features

Use the exact Request-191 feature family:
- Request-166 minute/path descriptors;
- Request-171 rich one-second aggregates;
- Request-173 market-regime context.

Exclude all execution-reference columns and oracle/future labels.

## Frozen models

Fit on historical fit rows:
1. HGB classifier for delayed_future_cost_coverable;
2. HGB regressor for delayed_future_best_base_return_pct.

Settings:
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- min leaf 75;
- L2 2.0;
- equal total sample weight per trading day;
- seeds 20261096 / 20261097;
- regressor target winsorization 0.5% / 99.5%.

Chronological calibration only:
- classifier selection gate = 75th percentile of calibration probability;
- regressor receives only an equal-day residual-mean offset.

No threshold may be fit on June 23-29.

## Required diagnostics

Row/state level:
- ROC AUC and average precision;
- value Spearman;
- delayed cost-cover prevalence;
- frozen selected rate;
- selected delayed-positive rate;
- selected delayed-best mean;
- uplift over all shadow states.

Episode-level recurrent bridge:
- traverse each BUY episode chronologically from minute 1 to 10;
- take the first state whose probability clears the frozen P75 gate and whose
  predicted delayed-best BASE return is >0;
- if no state qualifies, ABSTAIN for this diagnostic;
- report qualifying episode rate, first-entry minute distribution, realized
  delayed-best BASE mean, positive-opportunity rate, and per-day means.

This is still oracle-opportunity observability. It does not choose an exit.

## Frozen development bridge

Pass only if all hold:

1. state-level AUC >=0.60;
2. state-level value Spearman >=0.15;
3. value Spearman is positive on at least 4/5 days;
4. first-qualifying episode rate is between 10% and 70%;
5. first-qualifying delayed cost-cover rate >=60%;
6. first-qualifying delayed-best BASE mean >0;
7. first-qualifying delayed-best mean exceeds all shadow-state mean by >=+0.50pp;
8. first-qualifying delayed-best mean is positive on at least 4/5 days.

A pass licenses Request 194: an honest WAIT / ENTER / ABSTAIN controller with
recurrent post-entry exit management and the KRW 1,000,000 reference ledger.
