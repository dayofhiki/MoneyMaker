# Request 172 — uncertainty-aware rich-second patience result

Authoritative run: `36033838481`  
Artifact: `10823888192`

No new market dates were opened.

## Decision

**FAIL**

The fit/calibration-only 20th-quantile lower bound achieved reasonable
one-sided coverage but collapsed the patience set.

- evaluation lower-bound coverage: 79.11%;
- selected rows: 29 / 5,662 = 0.51%;
- selected realized excess mean: +1.1251%;
- selected positive-target rate: 58.62%;
- selected day-balanced excess: +0.4411%;
- bootstrap 95% interval: [-1.6944%, +2.2567%];
- positive selected-excess days: 3/5.

Daily:
- Jun 8: 0 selected rows;
- Jun 9: 3 selected, mean -2.7054%;
- Jun10: 8 selected, mean +2.9477%;
- Jun11: 12 selected, mean +1.3383%;
- Jun12: 6 selected, mean +0.1837%.

The conservative lower-bound formulation is therefore too sparse and unstable
to support an executable recurrent patience controller.

## Research consequence

Do not search the quantile or relax the zero boundary on June 8-12.

The next branch should add a distinct causal information axis rather than
further thresholding the same ticker-local value estimate. In particular,
attach market-wide momentum regime/breadth information at every POSITION state:
runner breadth, cross-sectional momentum, top attention intensity and market
activity. A continuous market-scanning trader has access to this context
causally, and the current POSITION controller largely does not.
