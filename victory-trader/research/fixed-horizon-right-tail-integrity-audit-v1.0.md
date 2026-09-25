# Request 196A — fixed-horizon right-tail integrity audit

## Purpose

This is a design-integrity audit, not a new trading strategy and not promotion
evidence.

Request 196 labels shadow states against an inherited HOT+30m cap. Therefore a
1-minute shadow state can have about 29 minutes of future opportunity while a
10-minute state has about 20 minutes. Because the causal feature family includes
clock/path-age features, the observed winner AUC could partly reflect remaining
label horizon rather than stock-specific entry quality.

Request 196A removes that shortcut.

## Data

Reuse only already-open development data:
- Request-171 fit POSITION rows for training;
- Request-163 matching historical scan for labels/regime;
- Request-178 Jun 23/24/25/26/29 for development evaluation.

No new dates are opened.

## Fixed label

For each causal shadow state with minutes_held in [1,10]:

- delayed entry = first executable minute open at the completed state boundary;
- future candidate exits = exact executable opens strictly after entry and no
  later than state_t + 20 minutes;
- states without the full 20-minute regular-session window are excluded;
- fixed20_best_base_pct = best BASE return among those exits;
- fixed20_strong_winner_3 = 1[fixed20_best_base_pct >= +3%].

The 20-minute window is fixed before evaluation because every 1-10m shadow state
can fit inside the original 30m research envelope away from session close.

## Models

Train on Request-171 FIT only. Do not merge calibration into training.

1. Full current-state model: Request-195 causal feature family.
2. No-clock model: same features excluding minutes_held, minutes_since_open,
   and minutes_to_close.

No sequence model is needed because Request 196 already showed no sequence
increment.

## Shortcut diagnostics

On fresh development states report:
- full fixed-horizon AUC / average precision;
- no-clock fixed-horizon AUC / average precision;
- AUC from minutes_held alone and negative minutes_held alone;
- per-day full and no-clock AUC;
- fixed-horizon winner prevalence by shadow minute;
- top-quartile-by-full-score winner rate and actual fixed20_best_base_pct uplift.

Also report the old variable-horizon strong_winner_3 prevalence by shadow minute
for direct comparison when available.

## Interpretation

If fixed-horizon and no-clock AUC remain strong, Request 196's signal is not
primarily a remaining-horizon shortcut.

If AUC collapses materially, downgrade the Request-196 right-tail conclusion
and redesign the entry target before any executable policy.
