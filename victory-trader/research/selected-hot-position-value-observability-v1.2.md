# Selected-HOT position value observability v1.2 — request 145

## Status

Frozen after the recurrent-trader implementation audit and before inspecting
any output from this corrected branch.

Request 144 is incomplete and produces no research evidence. Its first attempt
was terminated by a runner shutdown. Its rerun preserved the original request
head SHA and therefore did not include later performance fixes. No request-144
result artifact exists.

The implementation audit also found two semantic defects before any position
result was observed:

1. a cross-session lag boundary could contaminate multi-day batch features;
2. the next executable minute open was exposed as a model input even though it
   occurs after the HOLD/EXIT decision.

Those defects are corrected here. Request 145 is a new no-new-date diagnostic,
not a reinterpretation of request 144.

## Dates

No new market session may be opened.

- fit position paths: 2026-04-30, 05-01, 05-04, 05-05, 05-06;
- calibration position paths: 2026-05-07, 05-08, 05-11, 05-12, 05-13;
- evaluation position paths: 2026-05-14, 05-15, 05-18, 05-19, 05-20.

May 21 and later remain sealed.

## Entry selector

Request 140B remains a formal failure and is not promoted.

Its fresh ordering evidence may be reused because pooled and per-day AUC/value
ordering gates passed. However the historical request-140B diagnostic
top-quartile threshold was calculated only after future-labeled calibration rows
were retained.

For request 145:

- classifier training remains on economically labeled FIT rows;
- the probability threshold is frozen at the 75th percentile across **all**
  CAL feature rows, regardless of whether a future economic label exists;
- the threshold is never recomputed on evaluation dates.

This removes future label availability from policy selection.

## Causal POSITION state

At timestamp `state_t`, the completed minute ending at `state_t` and second
bars completed no later than `state_t` are observable.

Allowed current-price state uses the completed bar close.

The open of the minute beginning at `state_t` is the next executable exit
reference. It is forbidden from MODEL_FEATURES and may be used only for
execution labels/accounting.

Therefore the causal path features are:

- frozen broad minute/state features;
- frozen trailing completed-second features;
- minutes held;
- log entry price;
- log completed current close;
- entry-to-completed-close return;
- running completed-bar high/low return;
- drawdown from completed running high;
- recovery from completed running low;
- minutes since regular-session open;
- minutes to regular-session close.

No next open, exit-now return, next-minute return, future best exit, oracle
value, or post-decision second bar may enter model inputs.

## Semantics-preserving performance changes

The following changes affect implementation cost only:

- second windows use binary search for the exact original completed-second
  interval;
- sessions are streamed one day at a time rather than concatenated;
- only anchored ticker-days are materialized after market-wide rank is computed;
- running high/low are updated incrementally;
- future BASE exit labels use a precomputed suffix maximum;
- session-clock conversions are cached by timestamp.

Regression tests freeze the second-window boundaries and shared BASE execution
cost calculation.

## Diagnostic A — one-minute continuation

Unchanged target:

    hold_advantage_1m_pct =
        BASE return from next-minute exit
        - BASE return from exit at the next executable open after state_t

Model:

- HistGradientBoostingClassifier;
- learning_rate 0.05;
- max_iter 180;
- max_leaf_nodes 15;
- min_samples_leaf 75;
- l2_regularization 2.0;
- random_state 20261052;
- chronological Platt calibration.

Semantic HOLD diagnostic remains calibrated probability > 0.5.

## Diagnostic B — remaining option value

Unchanged target concept:

    remaining_option_value =
        best later BASE exit through the 30-minute entry cap
        - BASE exit available immediately after state_t

Fit-only held-minute baselines remove the trivial time-remaining effect before
regression.

Model:

- HistGradientBoostingRegressor;
- learning_rate 0.05;
- max_iter 180;
- max_leaf_nodes 15;
- min_samples_leaf 75;
- l2_regularization 2.0;
- random_state 20261053;
- fit-only 0.5%/99.5% target winsorization;
- one chronological calibration mean-bias offset.

Semantic patience diagnostic remains adjusted predicted excess value > 0.

## Explicit execution/path coverage

Report original evaluation anchors as the denominator, including anchors that
never produce a usable post-entry state.

Report separately:

- all first-HOT anchor-to-path coverage;
- policy-selected anchor-to-path coverage;
- full market rows processed;
- market rows materialized for position ticker-days;
- second-data coverage.

Missing paths are execution-risk evidence and are not silently deleted from
coverage.

## Frozen bridge

Request 145 passes only if the corrected policy-selected subset satisfies:

1. anchor-to-position-path coverage >= 90%;
2. HOLD AUC > 0.50;
3. semantic HOLD rows have positive arithmetic one-minute BASE advantage;
4. semantic HOLD rows have positive day-balanced one-minute BASE advantage;
5. remaining-value global Spearman > 0;
6. median eligible same-held-minute Spearman > 0;
7. predicted-excess-positive rows have positive day-balanced realized excess;
8. conditions 2 through 7 are simultaneously positive on at least four of five
   evaluation sessions with sufficient support.

Passing is an observability bridge only. It does not establish an executable
profitable trader.

If it passes, the next step is development replay of the full
ABSTAIN/WAIT/BUY -> HOLD/EXIT loop with explicit executability/liquidity state
before any May21+ fresh policy test.

May 21 and later remain sealed.
