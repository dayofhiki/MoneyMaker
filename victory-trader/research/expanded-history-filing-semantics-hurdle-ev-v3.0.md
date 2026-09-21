# Expanded-history filing-semantics hurdle EV v3.0 — pre-registration

## Status

Pre-registered after v2.9 completed and failed, before any v3.0 return result.
Use only September 2025 through March 2026. April 2026 and later remain sealed.

## Fixed motivation

v2.9's strictly-prior news counts and coarse sentiment did not improve the
v2.8 comparator. News existed within 72 hours for only 13.2% of evaluation
anchors, 49 of 56 selected trades overlapped v2.8, and the seven new selections
had a -2.25% BASE mean. Retire that news representation.

The independent historical 8-K access probe was run before this model and found
complete primary/secondary/tertiary taxonomy plus at least 40% 30-day filing
coverage in every sampled evaluation month. v2.8 used only broad primary
category counts. v3.0 tests whether point-in-time filing semantics distinguish
fresh equity supply and debt pressure from operating catalysts.

## Frozen universe, outcome, folds, and comparator

Use the exact v2.8 execution-feasible first-anchor universe, fixed 15-minute
BUY outcome, LIGHT/BASE/STRESS costs, chronological folds, model classes,
hyperparameters, calibration, seeds, and `EV > 0` boundary. The comparator is
the exact v2.8 split+supply hurdle model. Evaluation remains January, February,
and March 2026 only.

## Strictly-prior semantic enrichment

For trading day D, include only disclosures with `D-30 <= filing_date < D`.
Rows are deduplicated by accession number within each semantic bucket.

The buckets are fixed from the provider taxonomy before return evaluation:

- equity supply: public offering, private placement, underwriting agreement,
  acquisition-consideration shares;
- debt: secondary category `debt_activity`;
- operating catalyst: clinical trial results, preliminary/quarterly results,
  guidance update, partnership/collaboration, licensing agreement, strategic
  initiative;
- adverse: material litigation and executive/CFO/director departures.

Append exactly eleven derived features: log1p 30-day and 7-day accession counts
for each of the four buckets; log1p total semantic accessions; log1p age of the
latest semantic filing; and `(equity supply - operating catalyst) / (equity
supply + operating catalyst + 1)`.

No news feature, headline keyword, category discovered from January-March
returns, threshold, category gate, or return-conditioned subgroup is allowed.

## Policies and promotion

Evaluate buy-all, the exact v2.8 comparator, and the v3.0 filing-semantic model.
All use the same first anchor and 15-minute action. Promote only if v3.0:

1. has at least 15 evaluated trades in every month;
2. has arithmetic and day-balanced BASE mean above zero every month;
3. beats buy-all day-balanced BASE every month;
4. is not worse than v2.8 BASE in any month;
5. does not worsen BASE p05 or STRESS mean versus buy-all in any month;
6. has positive hurdle-EV Spearman every month;
7. has a positive day-cluster bootstrap 95% lower bound;
8. has complete, strictly-prior filing queries;
9. has positive BASE ledger marked return and complete accounting every month.

No April 2026+ data may be opened by this run. Failure retires this semantic
representation; its categories, windows, model, horizon, costs, or EV boundary
must not be tuned on the observed January-March results.
