# Request 166 — BUY-gated POSITION value observability

## Purpose

Request 165 validated a causal HOT -> BUY/WAIT/ABSTAIN bridge on June
15/16/17/18/22. The next question is whether the corrected POSITION state can
reliably distinguish HOLD from EXIT **conditional on the new BUY population**.

This request opens no new market dates. It deliberately keeps the corrected
Request-145 position feature/model family unchanged so the only substantive
change relative to that branch is the validated Request-165 entry gate.

## Frozen entry policy

Use the Request-163 economic-opportunity classifier and its policy-safe
calibration rule, plus the Request-165 executability classifier and threshold.

At first HOT:
- BUY iff economic gate passes and executability gate passes;
- WAIT iff economic gate passes and executability gate fails;
- ABSTAIN otherwise.

Only BUY anchors become POSITION paths.

## Development split

Already-opened dates only.

Position fit:
- 2026-04-30
- 2026-05-01
- 2026-05-04
- 2026-05-05
- 2026-05-06
- 2026-05-07
- 2026-05-08

Position calibration:
- 2026-05-11
- 2026-05-12
- 2026-05-13
- 2026-05-14
- 2026-05-15
- 2026-05-18
- 2026-05-19
- 2026-05-20

Later opened diagnostic evaluation:
- 2026-06-08
- 2026-06-09
- 2026-06-10
- 2026-06-11
- 2026-06-12

June 15+ Request-165 fresh sessions are not used to fit or tune this position
bridge.

## Frozen POSITION state and targets

Identical to corrected Request 145:

- completed minute and completed-second state only;
- entry-relative path memory;
- running high/low, drawdown/recovery and session clock;
- no next execution open in MODEL_FEATURES;
- one-minute HOLD advantage target;
- remaining multi-minute option-value target;
- fit-only held-minute baselines;
- unchanged HistGradientBoosting + Platt HOLD model;
- unchanged remaining-value regressor and calibration offset.

No fixed holding period is selected.

## Bridge gate

Reuse the corrected Request-145 bridge unchanged for the BUY-gated population:

1. BUY anchor-to-position-path coverage >= 90%;
2. HOLD AUC > 0.50;
3. semantic HOLD rows have positive arithmetic one-minute BASE advantage;
4. semantic HOLD rows have positive day-balanced one-minute BASE advantage;
5. remaining-value global Spearman > 0;
6. median eligible same-held-minute Spearman > 0;
7. predicted-excess-positive rows have positive day-balanced realized excess;
8. conditions 2-7 are simultaneously positive on at least 4 of the 5
   evaluation sessions with sufficient support.

Passing is still only a POSITION observability bridge. It does not prove a
profitable recurrent policy.

If this passes, Request 167 may compose the frozen signals into an actual
chronological BUY -> HOLD/EXIT trajectory and measure trade/account P&L before
opening any later fresh market block.
