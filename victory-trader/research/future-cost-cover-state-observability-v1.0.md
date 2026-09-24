# Request 191 — causal future cost-cover state observability

## Status

Pre-registered after Request 190 failed to identify 30-minute viable episodes
from the original HOT state, and before implementing or running Request 191.

The evidence now says:

- Request 187: the BUY population has a robust positive 30-minute oracle ceiling;
- Request 189: causal POSITION state strongly ranks relative exit quality;
- Request 190: original-HOT state does not reliably predict which whole episode
  will become cost-coverable.

Therefore Request 191 moves the viability estimate into POSITION, where the
trader has received additional causal path information.

No new dates are opened and no executable stopping rule is changed.

## Population

Historical fit/calibration:
- Request-171 BUY-gated rich-second POSITION rows.

Fresh evaluation:
- Request-178 scored POSITION rows on Jun 23/24/25/26/29.

Attach the same Request-173 market-regime state used in Requests 176-178.

## Absolute future-viability labels

The causal POSITION builder already records, for each state:

    best_future_base_return_pct

This is the best exact executable BASE return available strictly after the
current state and before the 30-minute safety cap, measured from the original
entry.

Define:

    future_cost_coverable =
        1[best_future_base_return_pct > 0]

This asks a direct live-relevant question:

**If I keep the position open from this state, does any future executable exit
within the cap still recover all modeled BASE costs?**

Future outcomes are labels only.

## Causal features

Use:
- Request-166 causal POSITION/path features;
- original one-second aggregate features;
- Request-171 rich one-second aggregate features;
- Request-173 market-regime features.

No future execution references or oracle labels enter model inputs.

## Models

Fit on historical fit rows:

1. HGB classifier for future_cost_coverable;
2. HGB regressor for best_future_base_return_pct.

Settings:
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- min leaf 75;
- L2 2.0;
- equal total sample weight per trading day;
- seeds 20261094 / 20261095;
- regressor target winsorization 0.5% / 99.5%.

Chronological calibration:
- classifier selection gate = 75th percentile of calibration probability;
- regressor may receive only equal-day-weighted residual-mean offset.

No fresh threshold fitting.

## Required diagnostics

Overall and by day:
- classifier ROC AUC / AP;
- regressor Spearman;
- future-cost-cover prevalence;
- frozen top-quartile selected rate;
- selected future-best BASE mean;
- selected future-cost-cover positive rate;
- uplift over all states.

Also report early-state diagnostics for minutes 1-10 separately, because an
economically useful controller must identify viability before the opportunity
has already passed.

## Frozen development bridge

Pass only if:

1. overall AUC >=0.60;
2. overall value Spearman >=0.15;
3. early-state (minutes 1-10) AUC >=0.60;
4. early-state value Spearman >=0.15;
5. at least 4/5 days have positive value Spearman;
6. frozen selected future-cost-cover rate exceeds all-state prevalence by
   >=10 percentage points;
7. selected future-best BASE mean exceeds all-state mean by >=+0.50pp;
8. selected future-best mean is positive on at least 4/5 days.

A pass licenses composition with Request-189's capture-percentile timing head.
It does not itself promote an executable policy.
