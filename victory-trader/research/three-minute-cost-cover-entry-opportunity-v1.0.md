# Request 185 — three-minute cost-cover entry opportunity bridge

## Status

Pre-registered after Request 184 failed its development gate and before
implementing or running Request 185.

Request 184 cleanly separated the two entry-economics components on fresh
June 23/24/25/26/29:

- execution-drag ranking transferred strongly (Spearman about +0.77);
- exact next-minute gross movement did not (ENTER about +0.02, WAIT negative);
- decomposed net-value ranking remained positive but the semantic zero boundary
  selected only three resolved trades and all-day economics stayed negative.

The next target therefore changes from an exact one-minute return to a
near-term opportunity target. The horizon is frozen at three minutes because
three minutes is already the established causal runner/Active horizon in the
attention research; it is not selected from Request-184 outcomes.

No new market dates are opened.

## Population and causal representation

Reuse:
- Request-171 BUY-gated historical fit/calibration POSITION episodes;
- Request-178 fresh BUY-gated episodes;
- Request-184 causal HOT-time representation, including completed one-second
  aggregates and the BASE cost proxy.

No post-entry POSITION feature may enter the entry model.

## Labels

ENTER_NOW opportunity:
- enter at the frozen original entry reference;
- inspect exact executable opens at holding minutes 1, 2 and 3;
- label = maximum BASE round-trip return available among those three exits.

WAIT_1M opportunity:
- commit no capital at the original BUY time;
- use minute-1 open as the delayed entry reference;
- inspect exact executable exits one, two and three minutes after delayed entry
  (original holding minutes 2, 3 and 4);
- label = maximum BASE round-trip return among those exits.

These are hindsight opportunity labels used only for observability research.
They are not executable exit rules and must not be reported as realized trader
returns.

## Models

Fit two independent HistGradientBoostingRegressors on historical fit rows:

1. ENTER three-minute opportunity value;
2. WAIT three-minute opportunity value.

Frozen settings:
- squared-error loss;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- minimum 75 samples per leaf;
- L2 regularization 2.0;
- equal total sample weight per trading day;
- seeds 20261088 and 20261089;
- fit target winsorization 0.5% / 99.5%.

Chronological calibration may add only one equal-day-weighted residual mean
offset per head. No fresh threshold fitting.

## Diagnostic action

At the HOT timestamp:

    V_ENTER = predicted ENTER 3m BASE opportunity
    V_WAIT  = predicted WAIT 3m BASE opportunity
    V_ABSTAIN = 0

Choose the largest value with conservative ties:

    ABSTAIN > WAIT_1M > ENTER_NOW

This diagnostic action means only "which near-term opportunity appears worth
monitoring/attempting". It does not claim the hindsight-best exit is executable.

## Required diagnostics

Report:
- entry and one-second feature coverage;
- fresh Spearman for ENTER and WAIT three-minute opportunity values;
- action split and selected rate;
- realized hindsight three-minute opportunity mean, day-balanced mean and
  positive rate for selected states;
- corresponding all-BUY opportunity baseline;
- selected-minus-all-BUY uplift;
- 10,000-sample trading-day bootstrap for selected day-balanced opportunity;
- positive selected-opportunity days;
- by action, day and current price bin.

## Frozen development gate

The bridge passes only if:

1. entry-state coverage >=90%;
2. one-second feature coverage >=90%;
3. both opportunity-value Spearman >= +0.10;
4. selected rate is 10% to 70%;
5. selected day-balanced three-minute BASE opportunity > 0;
6. selected positive-opportunity rate >=60%;
7. selected mean opportunity exceeds the all-BUY mean by >= +0.25pp;
8. selected trading-day bootstrap 95% lower bound >0;
9. selected mean opportunity is positive on at least 4/5 days.

Even a pass is not a trading-policy promotion. A pass only justifies building
an executable recurrent exit controller specifically on the cost-covering
near-term entry population before any untouched forward validation.
