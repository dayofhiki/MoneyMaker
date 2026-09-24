# Request 176 — market-regime hurdle expected remaining value

## Purpose

Request 175 showed that the Request-174 classifier retains some ordering but
chronological Platt calibration compresses semantic probabilities around the
base rate, making P>0.5 unusably sparse.

This does not imply positive-value probability is useless. It implies a fixed
50% probability threshold ignores payoff asymmetry.

Request 176 therefore estimates the **economic expected value** of patience
directly with a three-head hurdle decomposition.

No new market dates are opened.

## Data

Reuse already-opened Request-171 rich-second POSITION phases and the
Request-173 market-wide causal context:

- fit: Apr 30 through May 8;
- calibration: May 11 through May 20;
- evaluation diagnostic: June 8-12.

June 15+ is not used here.

## State

Frozen causal state:
- Request-166 MODEL_FEATURES;
- Request-171 rich one-second aggregate features;
- Request-173 market-regime features.

No new feature subset is selected on June.

## Target

Keep the fit-minute-baseline-adjusted multi-minute remaining value:

    excess_remaining_option_value_pct

This remains an observability target using the inherited 30-minute safety cap.
It is not itself an executable stopping policy.

## Hurdle decomposition

Fit three independent heads on the fit partition.

1. Probability head:
   - target: value > 0;
   - same HGB classifier and chronological Platt calibration as Request 174.

2. Positive magnitude head:
   - fit only rows with value > 0;
   - regress the signed positive value.

3. Non-positive magnitude head:
   - fit only rows with value <= 0;
   - regress the signed non-positive value.

Magnitude heads use:
- squared-error HGB;
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- min leaf 75;
- L2 2.0;
- fit-only 0.5% / 99.5% target winsorization;
- one class-conditional chronological calibration mean-bias offset.

At runtime:

    hurdle_EV =
        P(positive) * E[value | positive]
        + (1 - P(positive)) * E[value | non-positive]

Positive magnitude predictions are floored at zero.
Non-positive predictions are capped at zero.

The semantic patience boundary is exactly:

    hurdle_EV > 0

No probability threshold or EV threshold is tuned on June.

## Frozen diagnostic gate

Request 176 passes only if all hold on June 8-12:

1. market-context coverage >=95%;
2. hurdle-EV Spearman versus realized excess value >= +0.08;
3. EV-positive states are >=10% of evaluable states;
4. their arithmetic realized excess mean >= +0.20%;
5. their day-balanced realized excess mean is positive;
6. trading-day cluster bootstrap 95% lower bound is positive;
7. both daily EV ranking and selected realized excess are positive on >=4/5
   sessions.

Do not change these rules after seeing output.

Passing still does not prove executable profitability. It only justifies using
the frozen EV head inside a chronological one-minute-at-a-time recurrent
patience controller on already-opened data. A later unopened block is required
before promotion beyond diagnostics.
