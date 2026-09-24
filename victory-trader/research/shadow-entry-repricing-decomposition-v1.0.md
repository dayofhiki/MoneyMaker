# Request 195 — shadow-entry repricing decomposition

## Status

Pre-registered after Request 194 failed its frozen asymmetric-risk/reward development gate and before implementing or running Request 195.

Request 194 established two useful facts. A causal hard-stop overlay materially reduced the severe-loss tail, but the calibration-selected policy remained negative on every fresh day and was worse than minute-1 liquidation. Profit-taking almost never fired on fresh data. Therefore money management is useful but cannot by itself rescue the current entry economics.

Request 193 had already shown that 1-10 minute shadow states retain positive hindsight opportunity on average, while a direct model of delayed-entry net opportunity could not rank them. Request 195 now decomposes that new-entry problem rather than threshold-tuning it.

Core question:

**Can the model separately learn the remaining attainable price move and the execution drag at a shadow state, using explicit causal repricing/chase descriptors, and thereby rank new-entry surplus better than Request 193?**

## Data

Historical fit/calibration:
- Request-171 fit and calibration POSITION rows;
- Request-163 historical scan for exact executable opens and market regime.

Fresh development:
- Request-178 POSITION + scan shards on Jun 23/24/25/26/29.

No new dates are opened. Fresh dates remain development-only.

Use causal shadow states with minutes_held in [1, 10].

## Delayed-entry labels

For a completed shadow state t:
- delayed entry = next causally executable open after the completed state;
- future exits = later exact opens through original HOT + 30 minutes;
- among those exits choose the exit that maximizes delayed-entry BASE return.

For that economically best future exit define:

remaining_best_gross_pct = raw price return from delayed entry to that exit

remaining_best_drag_pct = remaining_best_gross_pct - remaining_best_base_pct

remaining_best_base_pct = modeled BASE return from delayed entry to that exit

Thus gross - drag exactly reconstructs the oracle delayed-entry BASE opportunity.

Delayed entry and future exit prices are labels only. They are never model inputs.

## Causal repricing features

Start from the Request-191 feature family:
- causal minute/path state;
- rich completed one-second aggregates;
- market regime.

Add only causal shadow-path descriptors:
- 1-minute move change in entry_to_current_close_pct;
- 3-minute move change;
- 5-minute move change;
- 1-minute acceleration;
- running range width = running_max_return_pct - running_min_return_pct;
- absolute move already consumed since HOT;
- causal zero-move BASE cost proxy computed from the completed current close.

The original HOT entry is only a price anchor. No paid original-entry P/L, delayed execution price, future price, or oracle label is an input.

## Frozen models

Train two independent equal-day-weighted HGB regressors:
1. remaining_best_gross_pct;
2. remaining_best_drag_pct.

Settings for both:
- learning rate 0.05;
- 180 iterations;
- max leaves 15;
- min leaf 75;
- L2 2.0;
- target winsorization 0.5% / 99.5%;
- seeds 20261098 / 20261099.

Chronological calibration supplies only an equal-day residual-mean offset for each head.

At scoring time:
- predicted drag is clipped at >=0;
- predicted new-entry surplus = predicted gross - predicted drag;
- semantic action boundary is predicted surplus >0;
- no fresh percentile or threshold tuning is allowed.

## Required diagnostics

State level:
- gross Spearman;
- drag Spearman;
- reconstructed net-surplus Spearman versus actual remaining_best_base_pct;
- selected rate under predicted surplus >0;
- selected positive-opportunity rate;
- selected actual BASE opportunity mean;
- all-shadow actual BASE opportunity mean;
- selected-minus-all uplift;
- per-day net-surplus Spearman.

Episode bridge:
- traverse minutes 1-10 chronologically;
- take the first state with predicted new-entry surplus >0;
- otherwise ABSTAIN;
- report qualifying episode rate, first-entry minute distribution, actual delayed best BASE mean, positive-opportunity rate, uplift over all shadow states, and daily means.

This remains an observability/oracle-opportunity bridge. It does not claim executable profit because the eventual exit is hindsight-best.

## Frozen development gate

Pass only if all hold:
1. gross Spearman >=0.15;
2. drag Spearman >=0.50;
3. reconstructed net-surplus Spearman >=0.15;
4. net-surplus Spearman positive on at least 4/5 fresh days;
5. first-qualifying episode rate between 10% and 70%;
6. first-qualifying positive-opportunity rate >=60%;
7. first-qualifying actual BASE opportunity mean >0;
8. first-qualifying mean exceeds all-shadow mean by >=+0.25pp;
9. first-qualifying mean is positive on at least 4/5 days.

A pass licenses an honest WAIT / ENTER / ABSTAIN controller that uses this decomposed surplus before paying entry cost, followed by the risk overlay and recurrent position management.

A failure means the bottleneck is not merely cost calibration: the current aggregate causal representation still cannot identify remaining upside from the new buyer's price basis, and the next architecture change must target richer sequence/price-action representation or a different entry population rather than more thresholds.
