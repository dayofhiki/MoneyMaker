# Corrected first-HOT opportunity revalidation v1.0 — request 148

## Purpose

Request 147 changes only the corrected Focus -> active transport utility. This
request asks whether the previously observed first-HOT economic-opportunity
signal survives on that corrected population.

No new market date is opened.

## Isolation principle

To avoid mixing two changes at once:

- upstream attention / active / HOT population uses request-147 actionable
  transport;
- request-140B ENTRY_FEATURES are unchanged;
- classifier/regressor hyperparameters are unchanged;
- economic labels and BASE execution-cost model are unchanged;
- the classifier selection threshold remains policy-safe and is frozen on all
  calibration feature rows, regardless of future label availability.

Therefore any change versus request 140B is attributable primarily to the
corrected upstream population rather than a new ENTRY feature set.

## Frozen dates

- opportunity fit: 2026-04-30, 05-01, 05-04, 05-05, 05-06;
- calibration: 2026-05-07, 05-08, 05-11, 05-12, 05-13;
- evaluation: 2026-05-14, 05-15, 05-18, 05-19, 05-20.

These dates are already opened by prior diagnostics. May 21 and later remain
sealed.

## Causal population

Market inference uses all current pre-runner rows. Future next-minute bar
existence is never used to decide whether a current ticker is ranked.

Active priority is:

    market_hazard_probability
    * next_minute_observability_probability

Stage-2 target treats absence of an exact next-minute bar as no exact
next-minute crossing. No future target is an input feature.

## Frozen opportunity model

Use the exact request-140B ENTRY_FEATURES and model families.

Training economic labels may use future outcomes. Missing future economic
outcomes remain unknown and are not converted into negative labels.

The policy classifier threshold is the 75th percentile of classifier
probability across all calibration feature rows, including rows without a
future economic label.

## Gate

Use the original opportunity-learning gate:

1. at least 100 labeled evaluation rows per day;
2. economic label coverage >= 95% on every evaluation day;
3. pooled classifier ROC AUC >= 0.55;
4. AUC > 0.50 on at least four of five evaluation days;
5. pooled value Spearman >= 0.05;
6. Spearman > 0 on at least four of five days;
7. selected oracle BASE mean > 0 and > all-HOT mean;
8. selected positive rate exceeds all-HOT positive rate by >= 5 percentage
   points;
9. selected mean is nonlower than all-HOT mean on at least four of five days.

Passing revalidates economic-opportunity ordering only. It does not prove an
executable profitable trader.

## Next step

If request 148 passes, add a separate causal executability/liquidity value to
the ENTRY controller and develop BUY vs WAIT vs ABSTAIN action value.

If request 148 fails only on economic-label coverage while ordering remains
positive, retain the ordering evidence but move executability into a separate
explicit gate rather than manipulating the economic target.

May 21 and later remain sealed.
