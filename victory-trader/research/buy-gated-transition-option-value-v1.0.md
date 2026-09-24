# Request 168 — transition-enhanced remaining-value observability

## Purpose

Requests 166-167 reject the binary one-minute HOLD target on the current
Request-165 BUY population. The surviving signal is multi-minute remaining
economic value.

Request 168 therefore changes the **information set of the remaining-value
regressor only**. It does not tune the failed one-minute HOLD classifier, a
holding horizon, a probability threshold, or the entry policy.

## Data

No new market date is opened.

Reuse Request-166 BUY-gated position rows:
- fit: Apr 30 through May 8;
- calibration: May 11 through May 20;
- evaluation diagnostic: June 8-12.

June 15+ remains out of this diagnostic.

## Target

Unchanged from corrected Request 145/166:

    remaining_option_value =
        best later BASE exit inside the inherited 30-minute safety cap
        - EXIT-now BASE return

Subtract the fit-only held-minute baseline before regression.

This label is an oracle observability target. It is not an executable exit rule.

## Only intervention

Start with the Request-166 option model information set and add exactly the
Request-167 causal position-path representation:

Current path:
- entry-relative return;
- maximum favorable excursion;
- maximum adverse excursion;
- drawdown from running high;
- rebound from running low;
- running path range;
- minutes since running high;
- minutes since running low;
- observed-minute fraction;
- path efficiency.

Exact timestamp deltas for each path variable:
- 1m, 2m, 3m, 5m, 8m, 13m.

Missing timestamps remain missing. No gap compression.

The regressor family, capacity, target winsorization, chronological
calibration-offset method and random seed remain identical to the baseline
option regressor.

## Frozen promotion gate

Request 168 passes only if all conditions hold on June 8-12:

1. pooled Spearman improves over the unchanged baseline by at least +0.02;
2. median same-held-minute Spearman improves by at least +0.01;
3. predicted-positive states are at least 10% of eligible states;
4. their day-balanced realized excess remaining value is positive;
5. trading-day cluster bootstrap 95% lower bound for their realized excess is
   positive;
6. both daily Spearman and selected realized excess are positive on at least
   4/5 sessions.

Do not weaken these conditions after inspecting the result.

Passing is still only an observability bridge. If it passes, the next experiment
must construct an executable recurrent stopping policy on already-opened data,
using no oracle future exit in action selection, before any new fresh block is
opened.
