# Request 173 — market-regime POSITION value

## Purpose

Requests 166-172 suggest ticker-local post-entry state contains a weak but real
remaining-value signal, while executable stopping targets remain unstable.
Request 171 improved robustness modestly, but Request 172 showed that simply
becoming more conservative collapses support.

A continuously scanning trader also observes the rest of the market while
holding a position. Request 173 tests whether same-minute market-wide momentum
context is a missing causal state variable.

No new market dates are opened and no new external data is fetched.

## Data

Reuse:
- Request-171 rich-second POSITION phase artifacts for Apr30-May20 and Jun8-12;
- Request-163 already-opened market-wide minute scan.

Only scan rows from the POSITION phase dates are used.

## Causality

For a POSITION state at timestamp t, market context is computed only from the
market-wide scan row at the same completed minute t.

No later minute and no position outcome enters a market-regime feature.

## Frozen market context

Add all 16 features together, with no post-result selection:

- market universe count;
- fraction of names >=5%, >=10%, >=20% from previous close;
- cross-sectional median, p90, p99 and dispersion of return from previous close;
- fraction of names with positive minute body;
- fraction with positive one-minute return;
- top-1, top-10 mean and p90 attention score;
- log total minute volume;
- log total minute transactions;
- top-10 volume concentration share.

The Request-171 ticker-local rich-second and original POSITION features remain.

## Target/model

Keep Request-166 excess remaining-option target unchanged.

Use the same HGB regression family/capacity and calibration method as the
Request-171 point model:
- squared error;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- min leaf 75;
- L2 2.0;
- fit-only 0.5%/99.5% winsorization;
- one chronological calibration mean-bias offset.

## Frozen development promotion gate

Relative to the Request-171 rich-second point model on June 8-12, pass only if:

1. market context is available on >=95% of evaluation states;
2. pooled Spearman improves by >=+0.02;
3. median same-held-minute Spearman improves by >=+0.01;
4. predicted-positive states are >=10% of eligible states;
5. selected day-balanced realized excess is positive;
6. selected trading-day bootstrap 95% lower bound is positive;
7. daily Spearman and selected excess are both positive on >=4/5 days.

Do not relax these criteria or select market features after inspecting June.

Passing is still only a development observability bridge. A later experiment
must compose an executable recurrent policy and ultimately validate on an
unopened block with BASE/stress costs and account replay.
