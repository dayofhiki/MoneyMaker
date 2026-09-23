# HOT promotion-episode economics v1.1 — pre-registration

## Status

Request 140 is a structural diagnostic after request 139.

It reuses only the already-opened request-138/request-139 sessions:
2026-05-07, 2026-05-08, 2026-05-11, 2026-05-12 and 2026-05-13.

No date after 2026-05-13 may be opened.

## Motivation

Request 139 evaluated only the first HOT state of each ticker-day. That is a
useful conservative bridge but it does not match the intended live trader:
a ticker may leave HOT and later be promoted again after new evidence arrives.

Request 140 therefore treats every causal transition into HOT as a distinct
promotion episode.

## Frozen episode definition

A HOT promotion is a row from the promoted request-138 runtime trace with:

- state == HOT;
- reason == learned_hot_promote.

Continuous retained HOT rows are not new episodes.

For each ticker-day, promotions are numbered 1, 2, 3, ... in time order.
Promotion 2+ is called a re-promotion.

## Frozen economics

Use the same request-139 labeling and execution assumptions:

- entry reference = exact next minute-bar open after promotion;
- no delayed fallback for a missing entry reference;
- fixed 1/2/5/10/15/30-minute outcomes;
- existing LIGHT/BASE/STRESS costs unchanged;
- 30-minute hindsight best BASE exit is diagnostic only.

## Required diagnostics

Report:

1. total promotion episodes and entry-reference coverage;
2. fixed-horizon and 30-minute oracle economics for all promotion episodes;
3. first-promotion versus re-promotion oracle economics;
4. promotion-index groups 1, 2 and 3+;
5. re-promotion spacing distribution;
6. Spearman correlation between the frozen HOT score and 30-minute oracle
   BASE return;
7. descriptive HOT-score quintiles with episode count, score mean, oracle
   mean/median and oracle-positive rate.

The score quintiles are descriptive only. No threshold, cutoff or policy may
be selected from them.

## Interpretation

This experiment does not promote a trading policy.

- If re-promotions have materially different economics from first promotions,
  subsequent ENTRY research must model promotion episodes rather than collapse
  a ticker-day to its first HOT state.
- If the frozen HOT score weakly or non-monotonically orders economic outcomes,
  ENTRY/ABSTAIN must use a separate economic-value model rather than treating
  the surge-detection score as an entry score.
- If all promotion episodes remain economically weak even under the
  non-executable oracle ceiling, the candidate/HOT definition must be revisited
  before opening a fresh ENTRY validation block.
- If a meaningful profitable tail exists but broad entry remains negative,
  proceed to a separately preregistered selective ENTRY/ABSTAIN model.

May 14 and later remain sealed.
