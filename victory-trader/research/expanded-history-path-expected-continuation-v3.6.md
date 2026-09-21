# Expanded-history path-aware expected continuation value v3.6 — pre-registration

## Status

Frozen after v3.5 January established that the all-month v3.5 bridge criterion
cannot pass, and before inspecting the v3.5 March outcome. Use only September
2025 through March 2026 development data. April 2026 and later remain sealed.

This experiment changes the one-step target/objective only. It does not search
new features, holding horizons, probability thresholds, market phases, or
feature subsets.

## Motivation fixed before v3.5 March

v3.5 added causal position-path memory to the EXIT-now versus HOLD-one-minute
diagnostic. January improved materially versus v3.4 but still failed its frozen
bootstrap condition; February passed all frozen conditions.

The remaining conceptual mismatch is that v3.4-v3.5 optimize the probability
that one more minute has positive incremental return. Trading utility depends on
magnitude as well as sign. A state with many tiny positive next minutes and a
smaller number of large negative next minutes can have P(HOLD wins) > 0.5 while
having negative expected incremental value.

v3.6 therefore asks a narrower question: can the exact same causal state predict
the arithmetic expected BASE incremental value of HOLD one more minute well
enough to select positive-value HOLD decisions?

## Frozen data, state and timing

Everything below is inherited unchanged from corrected v3.5:

- exact v3.2 first-anchor panels from request 84;
- frozen 2025 state panels from request 54 and frozen 2026 state panels from
  request 8;
- strict past-only fit/calibration/evaluation day partitions;
- unchanged v3.3 opportunity head and frozen P(opportunity) > 0.5 stratum;
- conceptual entry at the exact next-minute open after the frozen anchor;
- follow-up decisions at completed holding minutes 1-29;
- EXIT now uses the exact decision open and HOLD uses the exact next-minute open;
- no missing-timestamp fallback;
- BASE future sell friction only; entry cost is sunk;
- the exact v3.5 causal feature frame, including its ten entry-relative
  position-path features;
- April 2026+ remains sealed.

No new market, filing, news, quote, short-interest, supply, split or hand-built
interaction feature is allowed.

## Frozen regression target

Target is the unchanged v3.5 one-step arithmetic incremental advantage:

    hold_advantage_pct = (HOLD sell fill / EXIT sell fill - 1) * 100

Fit only rows with exact EXIT and HOLD references.

For each fold, calculate the 0.5th and 99.5th percentiles of the fit target only
and winsorize fit targets to those bounds. Evaluation targets are never clipped.

Fit one HistGradientBoostingRegressor with:

- loss = squared_error;
- learning_rate = 0.05;
- max_iter = 180;
- max_leaf_nodes = 15;
- min_samples_leaf = 75;
- l2_regularization = 2.0;
- random_state = 20261042.

These settings mirror the v3.5 capacity while changing only the objective from
positive-return classification to arithmetic expected-value regression.

## Frozen calibration and selection-bias allowance

On the chronological calibration partition:

1. predict raw expected advantage;
2. calculate one global additive calibration bias as
   mean(actual - raw prediction), and add it to every prediction;
3. on calibration rows whose bias-adjusted prediction is > 0, compute for each
   trading day the mean optimism = prediction - actual;
4. subtract from every prediction
   max(0, mean(daily optimism) + 1.645 * SE(daily optimism));
5. if fewer than five calibration trading days contain a positive selected row,
   disable HOLD for that fold by assigning adjusted expected value -infinity.

This selected-tail allowance is reused from the earlier v1.8 research
infrastructure. It is fixed here before v3.5 March is inspected and may not be
relaxed after results.

The diagnostic action is HOLD iff adjusted expected advantage > 0; otherwise
EXIT. There is no tunable probability or EV threshold.

## Required diagnostics

For every January-March evaluation fold report all-anchor and frozen
opportunity-gate strata:

- candidate/evaluated rows and exact-reference coverage;
- raw and adjusted predicted-value mean;
- Pearson and Spearman association with realized one-step advantage;
- HOLD selection rate and selected row count;
- always-HOLD incremental mean;
- selected policy incremental mean, assigning EXIT zero;
- day-balanced selected policy mean;
- mean selected predicted value and realized value;
- calibration bias, selected calibration days, mean daily optimism and frozen
  one-sided correction;
- the same policy summaries by held-minute bucket and market phase;
- 10,000-sample trading-day cluster bootstrap for opportunity-gate overall
  selected policy value;
- strict fold provenance.

Repeated rows remain counterfactual diagnostics, not an executable trajectory.

## Frozen bridge criterion

A separately preregistered recurrent ENTRY -> HOLD/EXIT trajectory experiment
is permitted only if, in every one of January, February and March, the frozen
opportunity-gate overall stratum satisfies all five conditions:

1. adjusted expected-value Spearman > 0;
2. selected policy incremental mean > 0;
3. day-balanced selected policy mean > 0;
4. exact-reference coverage >= 90%;
5. trading-day bootstrap 95% lower bound > 0.

Zero selected HOLD rows cannot pass conditions 1-3.

If v3.6 fails, do not tune the zero EV boundary, winsorization, calibration
allowance, horizon, held-minute subset, phase subset, model capacity or v3.5
path feature subset against January-March. The next branch must change the
state-transition formulation or candidate information source rather than
continue objective-threshold tweaking.
