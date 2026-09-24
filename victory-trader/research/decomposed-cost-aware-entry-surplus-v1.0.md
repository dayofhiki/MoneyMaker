# Request 184 — decomposed cost-aware entry surplus

## Status

Pre-registered after Request 183 failed all calibration-relative support gates
and before implementing or running Request 184.

Request 183 showed that ENTER_NOW and WAIT_1M value ordering transfers to the
fresh June 23/24/25/26/29 block, but increasingly selective percentile filters
still remained BASE-negative. Request 181 separately showed that the dominant
first-minute loss component is modeled execution drag rather than adverse raw
price movement.

Request 184 therefore changes the representation, not the threshold. It
decomposes entry economics into raw price movement and execution drag and adds
the causal one-second state available at the original HOT timestamp.

No new dates are opened. The Request-178 block remains development-only.

## Frozen population and labels

Use exactly the Request-171 historical BUY-gated fit/calibration POSITION
episodes for training and the Request-178 BUY-gated POSITION episodes for
development evaluation.

For every episode:

ENTER_NOW:
- gross target = raw entry-open to minute-1-open return;
- drag target = gross return minus frozen BASE round-trip return.

WAIT_1M:
- gross target = raw minute-1-open to minute-2-open return;
- drag target = gross return minus frozen BASE round-trip return.

Future prices are labels only.

## Causal entry representation

At the original HOT timestamp use only information already observable then:

- Request-182 minute/attention entry features;
- the fixed one-second aggregate features from the 60 seconds ending before
  the HOT timestamp;
- a deterministic BASE zero-move cost proxy computed from the current observed
  price.

No POSITION state after entry may enter the model.

The one-second window uses completed seconds only, ending at HOT timestamp - 1s.

## Four-head decomposition

Fit four independent HistGradientBoostingRegressors on historical fit rows:

1. ENTER gross;
2. ENTER execution drag;
3. WAIT gross;
4. WAIT execution drag.

Frozen settings:
- squared-error loss;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- minimum 75 samples per leaf;
- L2 regularization 2.0;
- equal total sample weight per trading day;
- seeds 20261084..20261087;
- fit-target winsorization at 0.5% / 99.5%.

Chronological calibration may add only an equal-day-weighted residual mean
offset to each head. No slope fit or threshold tuning.

Predicted execution drag is clipped at zero because the frozen execution model
cannot produce negative friction.

## Runtime action value

    V_ENTER = predicted_enter_gross - predicted_enter_drag
    V_WAIT  = predicted_wait_gross  - predicted_wait_drag
    V_ABSTAIN = 0

Choose the largest predicted value with conservative ties:

    ABSTAIN > WAIT_1M > ENTER_NOW

No percentile filter is allowed. The economic boundary remains semantic zero.

## Diagnostics

Report:
- historical and fresh causal entry-feature coverage;
- second-feature coverage;
- fresh Spearman for all four component heads;
- fresh Spearman for decomposed ENTER and WAIT net values;
- ENTER / WAIT / ABSTAIN rates;
- executed BASE mean and day-balanced mean;
- all-opportunity value including abstentions;
- positive-return and severe-loss rates;
- matched improvement versus current ENTER_NOW;
- 10,000-sample trading-day bootstrap of matched improvement;
- positive-return trading days;
- metrics by chosen action and entry-price bin.

## Frozen development gate

A candidate is worth later untouched validation only if all hold:

1. fresh entry-state coverage >= 90%;
2. fresh one-second feature coverage >= 90%;
3. ENTER and WAIT decomposed net-value Spearman are both positive;
4. executed rate is 3% to 40%;
5. executed-trade day-balanced BASE mean > 0;
6. all-opportunity day-balanced value > 0;
7. matched policy-minus-current day-balanced difference > 0;
8. matched-difference bootstrap 95% lower bound > 0;
9. severe-loss rate is no worse than current ENTER_NOW;
10. executed return is positive on at least 3/5 days.

Even a pass cannot promote on the already-opened Request-178 dates.

If ranking improves but economics remain negative, the next branch must alter
the economic opportunity target/horizon or obtain better execution information,
not tune percentiles on these dates.
