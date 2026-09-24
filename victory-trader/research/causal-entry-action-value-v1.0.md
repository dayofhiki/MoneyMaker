# Request 182 — causal ENTER / WAIT / ABSTAIN entry-value bridge

## Status

Pre-registered after Request 181 identified execution-dominant first-minute
loss and a robust positive one-minute WAIT effect, before implementing or
running Request 182.

No new market dates are opened. The Request-178 June 23/24/25/26/29 block is
development-only and cannot promote this policy.

## Question

Conditional on the already frozen Request-165 BUY population, can information
available at the original HOT/BUY timestamp choose among ENTER_NOW, WAIT_1M,
and ABSTAIN so that the selected executable action has positive BASE economic
value?

This experiment does not alter the upstream attention, economic-opportunity or
executability gates. It is an additional entry-timing/abstention layer.

## Causal entry state

For each BUY episode, join its original hot_t to the market scan row at that
same timestamp. Construct only causal entry-time features:

- frozen minute/attention features from BASELINE_FEATURES;
- cross-sectional attention rank computed at that timestamp;
- log current price;
- minutes since regular-session open;
- minutes to regular-session close.

No POSITION state after entry, future return, minute-1 observation or outcome
may enter the feature set.

## Action labels

Using historical BUY-gated POSITION paths only as labels:

### ENTER_NOW

Enter at the frozen original BUY entry open and exit at the exact minute-1
executable open.

Target = existing BASE round-trip return.

### WAIT_1M

Commit no capital at the original BUY time. Enter at the exact minute-1 open and
exit at the exact minute-2 open.

Target = BASE round-trip return under the unchanged execution-cost model.

### ABSTAIN

Value = exactly 0.

The one-minute exits are research instruments for isolating entry value. They
do not redefine the final trader's recurrent holding horizon.

## Training

Use only the old Request-171 chronological fit/calibration partitions.

Fit two independent HistGradientBoostingRegressors on the fit partition:

- one for ENTER_NOW BASE value;
- one for WAIT_1M BASE value.

Frozen settings for both:
- squared-error loss;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- minimum 75 samples per leaf;
- L2 regularization 2.0;
- equal total sample weight per trading day;
- seeds 20261082 and 20261083;
- fit-target winsorization at 0.5% / 99.5%.

Calibration may add only one equal-day-weighted residual mean offset per action.
No slope fit, probability threshold, action threshold, price cutoff or
fresh-June tuning is allowed.

## Runtime decision

At each original BUY event compute V_enter, V_wait and V_abstain=0, then choose
the action with the largest predicted value.

Ties resolve conservatively:

    ABSTAIN > WAIT_1M > ENTER_NOW

This gives a semantic zero economic boundary rather than a tuned selection
rate.

## Development evaluation

On Request-178 episodes:

- ENTER_NOW realizes exact minute-1 BASE return;
- WAIT_1M realizes exact minute-1 to minute-2 BASE return;
- ABSTAIN realizes 0 and commits no capital.

A selected action whose exact required reference is missing is unresolved, not
given an invented fill.

Comparator: frozen current behavior of ENTER_NOW followed by the minute-1 exit
for every auditable BUY episode.

## Required diagnostics

Report:

- entry-state feature coverage;
- ENTER / WAIT / ABSTAIN action rates;
- action-value coverage;
- predicted-vs-realized Spearman for both action heads;
- realized BASE mean and day-balanced mean among executed trades;
- policy value per BUY opportunity including zero-valued abstentions;
- positive-trade and severe-loss rates;
- matched policy-minus-current ENTER_NOW difference;
- 10,000-sample trading-day bootstrap for matched difference;
- metrics by chosen action and entry-price bin.

## Frozen development gate

A candidate entry layer is worth untouched forward validation only if:

1. entry-state feature coverage >=90%;
2. action-value coverage >=90%;
3. at least 10% and at most 80% of covered BUY opportunities are executed;
4. executed-trade day-balanced BASE mean >0;
5. all-opportunity day-balanced policy value >0;
6. matched policy-minus-current day-balanced difference >0;
7. matched-difference bootstrap 95% lower bound >0;
8. severe-loss rate is no worse than the current ENTER_NOW comparator;
9. both action-value heads have positive Spearman on the full development block.

Even a pass does not promote the policy because the June block is already open.
A pass only freezes the exact candidate for a later untouched block.
