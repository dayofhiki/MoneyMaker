# Dynamic State Policy v0.2

This policy is specified before viewing its corrected January-March results.

## Audit correction

v0.1 is invalid for strategy conclusions because signal scoring used the realized
next-minute open to estimate execution friction and score eligibility depended on
future label availability.

v0.2 fixes both issues:

- state scoring eligibility uses only the current observed state and current close;
- predicted base friction is estimated from the current close, never the next-minute open;
- future-return availability is used only to train labels, never to decide whether a
  held-out state may generate a signal;
- candidate ranking is completed before checking whether the selected ticker has an
  exact next-minute open. A missing open means the attempted entry is not filled;
  the policy does not substitute the second-best ticker using future knowledge.

## Purpose

Test whether a continuously monitored minute-state model becomes economically useful
when it behaves like a trader rather than a repeated fixed-horizon bet generator.

## Information set

Point-in-time State Trader v0.1 features only. No news, future bars, unsafe
fundamental fields, or holdout-month refitting.

For each held-out month:
- train 5-minute and 10-minute gross-return state models on the other two months;
- use the chronological final 20% of training days only for score calibration;
- score every decision-time-eligible post-+10% state in the held-out month.

## Policy

Capital:
- one long position maximum across the market;
- start each held-out month with equity 1.0.

Entry:
- rank states by predicted 10-minute gross return converted to an approximate
  base-net score using the current close;
- require score >= the training-calibration 99th percentile;
- choose the highest-scoring ticker at that minute;
- buy only if that chosen ticker has an exact next-minute open;
- do not fall through to another ticker if the chosen ticker cannot fill.

Hold / exit:
- while holding, re-score the held ticker each observed minute;
- hold while predicted 5-minute gross return is > 0;
- exit if predicted 5-minute gross return <= 0;
- also exit if price is >=5% below running HOD and below regular VWAP;
- hard maximum holding time 30 minutes;
- normal exits execute at exact next-minute open;
- if still holding at session end, liquidate at the last observed regular-session close.

Re-entry:
- allowed after a completed exit;
- no exit and new entry from the same decision minute.

## Reporting

Report:
- gross, light, base and stress trade returns;
- trade count and holding duration;
- base win rate, p05 and worst trade;
- compounded base monthly return;
- base maximum drawdown;
- worst day.

This remains development research on already-seen January-March 2026. It does not
consume December 2025 or formal validation/holdout data.
