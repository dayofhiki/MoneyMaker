# Request 171 — rich one-second POSITION microstructure proxies

## Purpose

Requests 166-169 show that the current post-entry state does not support a
robust executable HOLD/EXIT controller on the Request-165 BUY distribution.
Request 170 then established that historical tick trades are inaccessible under
the current Massive entitlement (HTTP 403).

Request 171 therefore adds genuinely new causal information derived only from
the still-accessible historical one-second aggregate feed.

No new market dates are opened.

## Data

Reuse Request-166 BUY-gated POSITION rows:
- fit: Apr 30 through May 8;
- calibration: May 11 through May 20;
- evaluation diagnostic: June 8-12.

For every POSITION state at t, one-second features may use only completed
second bars whose seconds end no later than t. No future second may enter.

## Target

Keep the surviving Request-166 target unchanged:

    excess remaining option value =
        best later BASE exit inside the inherited 30m cap
        - EXIT-now BASE return
        - fit-only same-held-minute baseline

This is an observability target, not an executable stopping rule.

## Frozen new information

Add all of the following together, with no post-result feature selection:

- 5s return, previous 5s return, 5s acceleration;
- 15s return, previous 15s return, 15s acceleration;
- 5s volume burst and transaction burst;
- last-5s share of 60s volume and transactions;
- tick-rule-like signed volume imbalance from one-second return signs;
- signed transaction imbalance;
- 60s directional return efficiency;
- one-second sign-flip/chop rate;
- seconds since local 60s high and low;
- current close location inside the 60s close range;
- current close versus 60s volume-weighted close;
- last-10s volatility relative to the preceding 50s;
- mean trade-size proxy in last 10s relative to preceding 50s.

The original Request-166 MODEL_FEATURES remain present. Historical tick trades
and NBBO are not used.

## Model

Keep the baseline option-value HGB regressor family/capacity unchanged:
- squared error;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- min leaf 75;
- L2 2.0;
- same random seed;
- fit-only 0.5%/99.5% target winsorization;
- one chronological calibration mean-bias offset.

## Frozen promotion gate

Request 171 passes only if all hold on June 8-12:

1. at least one rich second feature is available on >=85% of evaluation states;
2. pooled Spearman improves over baseline by >=+0.03;
3. median same-held-minute Spearman improves by >=+0.02;
4. predicted-positive states are >=10% of eligible states;
5. selected day-balanced realized excess remaining value is positive;
6. trading-day cluster-bootstrap 95% lower bound is positive;
7. daily Spearman and selected realized excess are both positive on >=4/5 days.

Do not weaken these gates after inspecting results.

Passing still does not prove profitability. If it passes, the next step is an
executable one-minute-at-a-time recurrent stopping policy on already-opened
data. If it fails, the current accessible one-second aggregate information is
insufficient and the next research branch should focus on regime conditioning
or data entitlement rather than more variants of the same feature family.
