# Request 163 — extended-history economic opportunity validation

## Why this request exists

Request 162 did not produce a model-quality result. It stopped before fitting
because the validated, more selective attention path produced only 1,292
labeled first-HOT rows in the original five-day fit block and 1,275 in the
five-day calibration block. The frozen safety floor was 1,500 labeled rows in
each block.

Request 163 does **not** lower that floor.

Instead it uses more already-opened historical sessions before the fresh June
evaluation. No June 8-12 outcome from Request 162 was inspected because Request
162 terminated before model fitting/evaluation.

## Frozen attention path

Unchanged from Requests 160-162:

- market-hazard fit through 2026-01-16;
- learned Focus cap 180;
- three-minute Active target;
- Active features = BASELINE_FEATURES + market_hazard_probability;
- no market_rank;
- Active threshold = 0.003093198572895277;
- dynamic Active population;
- HOT budget = 10;
- existing one-second HOT reranker family and fit window.

## Extended development history

All development dates below were already opened before this request.

Fit:
- 2026-04-30
- 2026-05-01
- 2026-05-04
- 2026-05-05
- 2026-05-06
- 2026-05-07
- 2026-05-08

Calibration:
- 2026-05-11
- 2026-05-12
- 2026-05-13
- 2026-05-14
- 2026-05-15
- 2026-05-18
- 2026-05-19
- 2026-05-20

Fresh evaluation remains exactly:
- 2026-06-08
- 2026-06-09
- 2026-06-10
- 2026-06-11
- 2026-06-12

June 1-5 remain Request-161 validation evidence and are not used here.

## Frozen support rule

Require at least **1,500 labeled first-HOT rows** in both fit and calibration.
If either block remains below 1,500, stop again. Do not lower the rule.

## Frozen economics and models

Unchanged from Request 140B / Request 148 / Request 162:

- first HOT per ticker-day;
- exact next-minute entry reference;
- same 30-minute hindsight opportunity target;
- same BASE cost assumptions;
- same ENTRY_FEATURES;
- same classifier hyperparameters;
- same regressor hyperparameters;
- same calibration-bias procedure;
- same calibration top-quartile selector;
- same promotion gate.

## Promotion gate

Require all original conditions:

1. >=100 labeled first-HOT rows per fresh day;
2. >=95% economic-label coverage per fresh day;
3. pooled classifier ROC AUC >=0.55;
4. AUC >0.50 on >=4/5 days;
5. pooled value Spearman >=0.05;
6. Spearman >0 on >=4/5 days;
7. selected oracle BASE mean >0;
8. selected oracle mean > all-HOT oracle mean;
9. selected positive rate >= all-HOT positive rate +5pp;
10. selected oracle mean >= daily all-HOT mean on >=4/5 days.

Passing means causal economic-opportunity ordering survived on fresh data. It
does not prove executable trading profitability.
