# Request 177 — equal-trading-day weighted hurdle EV

## Purpose

Request 176 produced the strongest current post-entry result but missed the
frozen promotion gate only because its five-day bootstrap lower bound remained
slightly below zero. The selected average was dominated by a few strong days.

Request 177 targets this structural weakness without using June outcomes to
change a threshold, feature set, target or policy boundary.

No new market dates are opened.

## Frozen state and target

Identical to Request 176:
- Request-166 causal POSITION state;
- Request-171 rich one-second aggregate features;
- Request-173 market-regime features;
- fit-minute-baseline-adjusted excess remaining option value;
- same three-head hurdle decomposition;
- semantic boundary hurdle_EV > 0;
- same model capacities and execution/cost assumptions.

## Only intervention

Within every training or calibration head, give every trading day equal total
sample weight.

For fit regression/classification:
- each row receives weight proportional to 1 / rows_in_its_day;
- normalize weights to mean one.

For Platt calibration:
- apply the same equal-day total weighting.

For positive and non-positive magnitude calibration offsets:
- compute residual mean inside each calibration day;
- use the arithmetic mean of those day means.

This change is fixed before June evaluation.

## Frozen diagnostic gate

Reuse Request-176 gate unchanged:

1. market-context coverage >=95%;
2. EV Spearman >= +0.08;
3. EV-positive states >=10% of evaluable states;
4. selected realized excess mean >= +0.20%;
5. selected day-balanced realized excess mean >0;
6. trading-day cluster bootstrap 95% lower bound >0;
7. both daily EV ranking and selected realized excess >0 on >=4/5 sessions.

Do not relax or retune after output.

Passing still does not prove executable profitability. It only permits the
frozen EV head to enter a chronological one-minute-at-a-time recurrent
trajectory experiment on already-opened data before later fresh validation.
