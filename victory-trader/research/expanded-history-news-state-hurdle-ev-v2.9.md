# Expanded-history strictly-prior news-state hurdle EV v2.9 — pre-registration

## Status

Pre-registered after v2.8 completed and failed, before implementation and before
viewing any v2.9 return result. Use only September 2025 through March 2026.
April 2026 and later remain sealed.

## Fixed motivation

The v2.8 split+supply model produced positive gross mean only in January
(+1.007%) and remained BASE-negative in every month. February and March were
gross-negative. Split history therefore did not replicate a continuation versus
exhaustion distinction. The next branch must add a genuinely different causal
state source rather than retune a threshold, cost, horizon, or model.

An earlier independent access probe (request 7) established that Massive news
queries are available without retries and that publication timestamps and
ticker-specific sentiment metadata are present. News coverage was sparse
(17.8% within 72 hours overall), so absence of provider-tagged news is retained
as an observed zero and is not interpreted as proof that no catalyst exists.

## Frozen universe, target, and folds

Keep the exact v2.8 execution-feasible first-anchor universe:

- first causal eligible anchor only;
- dollar_volume_5m >= 100,000 USD;
- transactions_5m >= 50;
- active_minute_fraction_15m >= 0.80;
- no fall-through to a later anchor;
- fixed 15-minute BUY outcome under the existing LIGHT/BASE/STRESS costs.

Evaluate only January, February, and March 2026 with the exact strictly-past
fit/calibration folds used by v2.8. No fresh validation month is consumed.

## Strictly-prior news enrichment

For every anchor at UTC timestamp T, query ticker-tagged articles over
[T-72 hours, T), using `published_utc` only. Articles published at or after T
must never enter a feature. Sentiment counts include only insights whose ticker
is empty or equals the anchor ticker; a co-mentioned ticker's sentiment is not
transferred.

Append exactly these eight features to the v2.8 split+supply frame:

1. news_any_6h;
2. log1p(news_count_6h);
3. log1p(news_count_24h);
4. log1p(news_count_72h);
5. log1p(minutes since latest prior article), missing when none;
6. log1p(positive ticker insights in 72h);
7. log1p(negative ticker insights in 72h);
8. (positive-negative)/(positive+negative+neutral+1).

No headline keywords, publisher filters, sentiment threshold, news-count gate,
or return-conditioned news subgroup may be introduced.

## Frozen models and policies

The comparator is the exact v2.8 split+supply three-head hurdle model. The
primary model uses the same classes, hyperparameters, seeds, winsorization,
chronological Platt calibration, and magnitude offsets, appending only the eight
news features. Both estimate:

    hurdle_EV = P(win) * E[gain | win, X]
                - (1 - P(win)) * E[loss | loss, X]

Executable policies are:

- feasible_earliest_15m_cap1: buy every feasible first anchor;
- split_supply_hurdle_ev_15m_cap1: exact v2.8 comparator, BUY iff EV > 0;
- news_split_supply_hurdle_ev_15m_cap1: primary, BUY iff EV > 0.

Zero is a semantic expected-value boundary and is not tuned. No timing, HOLD,
SELL, alternative horizon, sizing, or later-anchor freedom is allowed.

## Promotion rule

Do not open April 2026 or later unless the primary policy satisfies all:

1. at least 15 evaluated trades in each month;
2. arithmetic and day-balanced BASE mean > 0 in each month;
3. day-balanced BASE strictly exceeds feasible earliest in each month;
4. monthly BASE is not worse than the v2.8 comparator in any month;
5. BASE p05 and STRESS mean are not worse than feasible earliest in any month;
6. news-model hurdle-EV Spearman versus realized BASE > 0 in each month;
7. pooled trading-day bootstrap 95% lower bound > 0;
8. every news query completes and every included article is strictly prior;
9. BASE ledger marked return > 0 with complete accounting in each month.

A sparse tail, zero-trade month, missing outcome, or LIGHT-only profit cannot
satisfy promotion.

## Failure rule

Do not tune news windows, feature subsets, sentiment mapping, EV boundary,
capacity, costs, horizon, or winsorization on January-March after results. If
ordering remains positive but selected gross returns do not replicate, retire
this news representation and test a different causal state source. If gross
alpha replicates but BASE stays negative, execution-cost observability becomes
the next bottleneck.

## Completed result — request 78

The preregistered run completed successfully and failed promotion:

- January: 39 trades, gross +0.841%, BASE -0.192%, day-balanced -0.036%,
  bootstrap 95% CI [-1.861%, +1.999%].
- February: 12 trades, gross -1.463%, BASE -2.870%, day-balanced -2.456%,
  bootstrap 95% CI [-5.499%, -0.259%].
- March: 5 trades, gross -3.347%, BASE -4.309%, day-balanced -4.309%,
  bootstrap 95% CI [-9.234%, +0.173%].

Trade-weighted pooled BASE was -1.133%, worse than the v2.8 comparator's
-0.874%. Only 13.2% of evaluation anchors had a provider-tagged article within
72 hours. Forty-nine of 56 v2.9 selections overlapped v2.8, while the seven
v2.9-only selections averaged about -2.25% BASE. Probability AUC changed by
only +0.0016, +0.0001, and -0.0089 across the three months. The coarse news
representation is retired without tuning its windows, sentiment mapping,
feature subset, or decision boundary.
