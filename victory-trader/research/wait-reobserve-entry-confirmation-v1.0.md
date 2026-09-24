# Request 186 — WAIT then re-observe entry confirmation

## Status

Pre-registered after Request 185 failed its three-minute opportunity bridge and
before implementing or running Request 186.

Request 185 showed an asymmetric result on the fresh June 23/24/25/26/29
block:

- ENTER_NOW three-minute opportunity ranking did not transfer (Spearman near 0);
- WAIT_1M three-minute opportunity ranking retained positive ordering;
- choosing an action only from the original HOT state still selected a
  BASE-negative opportunity set.

A real continuous trader does not commit at the original timestamp to enter one
minute later. WAIT means no capital is committed, one minute of new market data
arrives, and the trader decides again. Request 186 tests exactly that missing
re-observation step.

No new dates are opened.

## Population

Start from the frozen Request-165/178 BUY-gated episodes.

At original HOT:
- take WAIT unconditionally for this diagnostic;
- commit no capital.

At the exact minute-1 causal state:
- re-observe the market;
- decide ENTER_NOW or ABSTAIN.

Episodes without an exact minute-1 causal state are coverage misses.

## Causal WAIT-state representation

Use only information observable at the minute-1 re-observation state:

- BASELINE_FEATURES;
- original one-second aggregate features;
- Request-171 rich one-second aggregate features;
- current market price;
- session clock;
- deterministic BASE zero-move cost proxy.

Explicitly exclude:
- entry_open;
- entry-to-current return;
- running position P&L;
- drawdown/recovery relative to an entry;
- any future execution reference.

Although these rows come from POSITION research artifacts, only market-state
columns that would exist even if no position had been opened are allowed.

## Target

If the model chooses ENTER at minute 1:

- entry reference = exact minute-1 executable open;
- inspect exact executable exits one, two and three minutes later;
- target = hindsight-best BASE round-trip return across those three exits.

This is the same WAIT three-minute opportunity concept from Request 185, but
now predicted using the new information that arrives during WAIT.

The target is an observability label, not an executable exit rule.

## Model

Fit one HistGradientBoostingRegressor:

- squared-error loss;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- minimum 75 samples per leaf;
- L2 regularization 2.0;
- equal total sample weight per trading day;
- seed 20261090;
- fit target winsorization 0.5% / 99.5%.

Chronological calibration may add only one equal-day-weighted residual mean
offset.

Runtime diagnostic:

    ENTER iff predicted WAIT-state 3m BASE opportunity > 0
    otherwise ABSTAIN

No percentile threshold or fresh tuning is allowed.

## Required diagnostics

Report:
- WAIT-state coverage;
- one-second/rich-second coverage;
- fresh opportunity Spearman;
- comparison with Request-185 original-HOT WAIT Spearman;
- selected rate;
- selected actual three-minute BASE opportunity mean/day-balanced mean;
- selected positive-opportunity rate;
- all WAIT-state opportunity baseline;
- selected-minus-baseline uplift;
- 10,000-sample trading-day bootstrap;
- positive selected-opportunity days;
- by day and price bin.

## Frozen development gate

Pass only if:

1. WAIT-state coverage >= 85%;
2. rich-second coverage >= 90%;
3. fresh Spearman >= +0.20;
4. fresh Spearman improves on the Request-185 original-state WAIT score by
   at least +0.03;
5. selected rate is 5% to 60%;
6. selected day-balanced opportunity > 0;
7. selected positive-opportunity rate >= 60%;
8. selected-minus-all-WAIT mean uplift >= +0.50pp;
9. selected bootstrap 95% lower bound > 0;
10. selected opportunity mean is positive on at least 4/5 days.

A pass justifies the next executable experiment: WAIT -> re-observe -> ENTER,
then recurrently HOLD/EXIT using only reached causal states. It does not
promote a trading policy on already-opened dates.
