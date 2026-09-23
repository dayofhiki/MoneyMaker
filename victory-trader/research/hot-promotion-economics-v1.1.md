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

## Result — request 140A

Authoritative run: `35828868215`. This diagnostic reused only the already-opened
2026-05-07 through 2026-05-13 sessions and opened no new date.

Across 9,597 HOT-promotion episodes, 9,509 had an evaluable 30-minute oracle
exit. Broad promotion entry remained economically weak: the overall oracle
BASE mean was -0.104% and only 33.23% of evaluable promotions had any
BASE-positive exit.

Promotion order mattered strongly:

| Promotion order | Episodes | Oracle BASE mean | Oracle-positive rate |
|---|---:|---:|---:|
| first | 2,091 | +0.339% | 40.70% |
| second | 1,297 | +0.209% | 41.17% |
| third or later | 6,209 | -0.315% | 29.12% |

All re-promotions pooled together averaged -0.225% because third-and-later
promotions dominate the count. Median spacing between re-promotions was seven
minutes (p25 4m, p75 17m).

The frozen HOT score nevertheless showed meaningful economic ordering across
all promotion episodes. Spearman correlation with 30-minute oracle BASE return
was +0.150. Score quintiles were monotonic:

| HOT-score quintile | Oracle BASE mean | Oracle-positive rate |
|---|---:|---:|
| Q1 | -0.616% | 22.45% |
| Q2 | -0.493% | 24.87% |
| Q3 | -0.211% | 32.72% |
| Q4 | +0.115% | 40.90% |
| Q5 | +0.685% | 45.22% |

### Decision

Do not treat every HOT promotion as a new equal-quality entry opportunity.
First and second promotions remain plausible economic candidates, while the
large third-plus population is materially weaker. Promotion index and spacing
are causal state and should become explicit context in later ENTRY research.

The monotonic HOT-score result also establishes that the surge-detection score
contains real economic ordering, but it is not sufficient as an entry rule:
even the highest quintile is oracle-positive in fewer than half of episodes.
Proceed with a separate economic-opportunity model rather than thresholding HOT
score directly.
