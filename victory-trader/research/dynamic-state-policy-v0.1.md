# Dynamic State Policy v0.1

This policy is specified before viewing its January-March results.

## Purpose

Test whether the minute-state model becomes economically useful when it behaves
more like a trader and less like a repeated fixed-horizon bet generator.

## Information set

Point-in-time State Trader v0.1 features only. No news, future bars, holdout-month
refitting, or unsafe fundamental fields.

For each held-out month:
- train 5-minute and 10-minute gross-return state models on the other two months;
- use the chronological final 20% of training days only for score calibration;
- score every observed post-+10% state in the held-out month.

## Policy

Capital / universe:
- one long position maximum across the entire market;
- start each held-out month with equity 1.0;
- while flat, inspect all active runner states at each minute.

Entry:
- use predicted 10-minute base-net score;
- require score >= the training-calibration 99th percentile;
- if multiple tickers qualify at the same minute, choose the highest score;
- decision is made after the state minute closes;
- buy at the exact next-minute open;
- missing exact next-minute open means no entry.

Hold / exit:
- while holding, evaluate the same ticker each observed minute;
- continue holding while predicted 5-minute gross return is > 0;
- exit if predicted 5-minute gross return <= 0;
- also exit on a failure state: >=5% below running HOD and below regular VWAP;
- hard maximum holding time: 30 minutes;
- all normal exits execute at the exact next-minute open;
- if still holding at regular-session end, force liquidation using the last
  observed regular-session close as the reference price.

Re-entry:
- allowed after a completed exit;
- do not exit and open a new position from the same decision minute.

Execution / risk reporting:
- report light, base, and stress modeled fills;
- base equity is compounded trade by trade using 100% notional because only one
  position can be open;
- report monthly return, max drawdown, trade count, win rate, p05 trade return,
  worst trade, and worst day.

## Interpretation

This is development research on already-seen January-March 2026.
It is not deployable and does not consume December 2025 or the formal
validation/holdout periods.

If this policy does not materially improve after-cost economics, do not tune its
thresholds on January-March. Diagnose entry/exit information and execution
realism first, then version the policy and use a fresh development month.
