# Request 168 — transition-enhanced remaining-value result

Authoritative run: `36030507352`  
Artifact: `10822160028`

No new market dates were opened.

## Decision

**FAIL**

Adding Request-167 path-transition features to the surviving multi-minute
remaining-value regressor made the signal materially worse.

| metric | baseline | transition |
|---|---:|---:|
| pooled Spearman | +0.0900 | +0.0431 |
| same-minute median Spearman | +0.0533 | +0.0418 |
| positive same-minute groups | 93.10% | 65.52% |
| selected realized excess mean | +0.5120% | +0.0341% |
| selected day-balanced excess | +0.5782% | +0.0783% |
| selected bootstrap 95% lower | -0.0685% | -0.1925% |
| days with positive Spearman + selected excess | 4/5 | 2/5 |

Pooled Spearman changed by -0.0469 and same-minute median by -0.0115, both in
the wrong direction. The preregistered material-improvement gate therefore
failed.

The result rejects further expansion of the same path-transition information
family on this current BUY distribution.

## Next research direction

Keep the validated Request-165 entry bridge frozen. Change the POSITION target,
not the threshold or path-transition feature family.

The next diagnostic should estimate exact medium-horizon incremental exit value
at several predeclared horizons (2m, 5m, 10m) relative to EXIT now, using only
the corrected causal Request-166 state. These are prediction horizons, not
fixed holding commitments: any eventual recurrent policy must re-evaluate every
minute and may exit earlier.

If multi-horizon values are observable, compose them into a one-minute-at-a-time
recurrent patience controller on already-opened data before any later fresh
block.
