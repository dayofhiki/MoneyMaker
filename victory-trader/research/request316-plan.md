# R316: event heterogeneity audit on frozen May HOT decisions

Preregistered before event acquisition or event-conditioned evaluation.
This branch is independent from the concurrent Work research track.

## Question

Test whether the frozen R311 May11-20 HOT evaluation problem is materially
heterogeneous with respect to point-in-time event context. The audit does not
train a new predictor, choose a specialist, retune a threshold, alter risk, or
open June/July-August data. It asks whether a single unconditional momentum
problem is a reasonable abstraction for the existing cohort.

The R311 N/Q/M scores, 828 identities, decision clocks, 650 complete labels,
BASE costs and -2.5% absorbing stop are frozen. No evaluation label may affect
event acquisition, event definitions, membership, or missingness handling.

## Scope limitation

The existing protocol excludes same-day split events from the original momentum
universe. Therefore this audit cannot by itself test the full "reverse-split
day" phenomenon. It must explicitly verify/report same-day split coverage and
treat zero/near-zero coverage as a population-scope limitation, not evidence
against reverse-split trading. A later structural-event cohort is required if
same-day reverse splits are scientifically important.

## Point-in-time event families

Event flags are multi-label; no case is forced into one exclusive cause.

Primary fixed flags:

1. `fresh_news_24h`: at least one Massive ticker-tagged news item with
   `published_utc` in [decision_t-24h, decision_t].
2. `prior_8k_7d`: at least one Massive classified 8-K disclosure whose
   filing date is strictly before the trading day and no more than 7 calendar
   days old. Same-day 8-K records are not treated as known because the current
   disclosure endpoint does not supply a filing timestamp precise enough for an
   intraday decision.
3. `recent_reverse_split_20d`: at least one Massive split record with
   `adjustment_type=reverse_split`, execution_date strictly before the trading
   day and within 20 calendar days.

Secondary descriptive flags are fixed before acquisition:
`fresh_news_1h`, `recent_news_7d`, `prior_8k_30d`,
`recent_reverse_split_5d`, plus same-day 8-K and same-day reverse-split counts
as timing/scope audits only.

`known_event_any` is the union of the three primary flags.
`none_known` means all three primary sources were acquired successfully and
none of the primary flags is present. It never means that no real-world cause
existed.

FINRA short volume/interest are excluded from R316 because publication-lag
semantics are not frozen here. Social/community data and manually researched
labels are excluded. No hindsight label such as "accumulation" is allowed.

## Acquisition

Use the existing R311 states artifact. Event queries are determined only from
ticker, trading_day and decision_t. News is queried per ticker over the union of
its required seven-day lookback windows. Classified 8-K disclosures and stock
splits are queried market-wide over the fixed cohort date range plus the maximum
30-day lookback when practical, then filtered to cohort tickers.

All raw event records used by the audit are checkpointed. Missing/failing
sources remain explicit. Event flags may not be inferred from price outcomes or
filled as false after an acquisition failure.

Massive's current split endpoint is `/stocks/v1/splits`; do not use the
deprecated reference-splits endpoint.

## Frozen diagnostics

Primary labeled support is the unchanged R311 complete-label support.

For the whole cohort and for each primary flag true/false group report:

- sampled / observed / complete-label counts, positives and positive days;
- equal-day/episode-weighted net5 prevalence;
- frozen N/Q/M pooled AUROC, AP, Brier and same-day AUROC when defined;
- frozen M-Q same-day AUROC increment when defined.

Report event overlap/signatures descriptively without selecting profitable
combinations.

For each primary family, compare flag versus complement with fixed ticker-day
and trading-day cluster bootstrap (1,000 draws; seeds 20265160/20265161):

- weighted net5 prevalence difference;
- M same-day AUROC difference;
- M-Q same-day AUROC-increment difference.

Invalid rank draws are counted rather than replaced. No p-value fishing or
subcategory search is permitted.

As a secondary mechanism diagnostic, compute fixed univariate directionality
for each already-frozen R311 MOMENTUM feature within the primary groups only
when both classes exist. This is descriptive and cannot promote a feature or
specialist.

## Interpretation gate

R316 does not promote a trading policy. It labels event heterogeneity
"materially supported" only when at least one primary event family has:

- at least 40 complete labels, at least 6 positives, and positives on at least
  3 trading days; and
- either a flag-versus-complement weighted prevalence ratio >=2.0 or <=0.5 with
  the ticker-day 95% bootstrap interval excluding 1.0, OR an absolute
  flag-versus-complement M same-day-AUROC difference >=0.10 with the ticker-day
  95% interval excluding 0, OR an absolute difference in M-Q same-day AUROC
  increment >=0.10 with the ticker-day 95% interval excluding 0.

If support is insufficient, report "underpowered", not homogeneous.
If no adequately supported family meets the fixed gate, event conditioning is
not justified by this audit and the current momentum research remains the
default path.

Even if the gate passes, the next step is first a simple event-context feature
or interaction test on frozen chronological training. Specialist/MoE routing is
not justified until that cheaper alternative is tested.

## Integrity checks

- exact 828 R311 identities and frozen score columns;
- no changes to R311 scores, labels, costs, stop, decision_t or model inputs;
- all news timestamps <= decision_t for point-in-time news flags;
- no same-day 8-K treated as point-in-time known;
- no future split execution used;
- multi-label event flags;
- explicit source-acquisition failures;
- no June HOLD or July-August access;
- no outcome-dependent event taxonomy or subgroup selection.

The output is a diagnostic artifact, not a backtest or deployment claim.
