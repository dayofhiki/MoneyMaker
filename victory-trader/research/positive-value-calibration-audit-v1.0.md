# Request 175 — positive-value calibration / prevalence drift audit

## Purpose

Request 174 retained weak ranking information but the chronological Platt
calibration produced almost no P>0.5 states on June 8-12.

This request changes no model, policy, threshold, feature, label, or date. It
reconstructs Request 174 and audits where the absolute probability semantics
break.

## Data

Already-opened only:
- fit: Apr 30-May 8;
- calibration: May 11-May 20;
- evaluation: June 8-12;
- Request-163 market-wide scan for the same dates.

## Required diagnostics

For fit, calibration, and evaluation report:
- positive-target prevalence;
- positive and non-positive target mean magnitude;
- raw classifier AUC;
- calibrated AUC;
- raw/calibrated probability mean and quantiles;
- fraction above 0.5;
- calibrated Brier score;
- the same metrics by held-minute buckets 1-3, 4-7, 8-15, 16-29.

Freeze calibration probability decile edges on the calibration partition and
report actual positive rate inside those same bins on calibration and June.

Also report June day-level metrics.

## Interpretation rule

This audit does not promote anything.

Flag:
- material base-rate drift if calibration-to-June positive prevalence changes
  by >=5 percentage points;
- ranking survival if June AUC >0.5;
- semantic-boundary sparsity if <5% of June rows exceed calibrated P=0.5.

If base rate / intercept semantics drift while ranking survives, the next
target family should estimate **economic expected value**, not tune a fixed
probability threshold on June. A predeclared hurdle model may combine:
P(value positive), conditional positive magnitude, and conditional non-positive
magnitude, all trained/calibrated before evaluation.
