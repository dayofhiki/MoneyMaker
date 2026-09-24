# Request 173 — market-regime POSITION value result

Authoritative run: `36034267142`  
Artifact: `10822979620`

No new market dates were opened.

## Decision

**FAIL strict promotion gate, but market-wide context materially improved ranking**

Market context coverage was 100%.

| metric | Request-171 rich-second | + market regime |
|---|---:|---:|
| pooled Spearman | +0.095393 | +0.112865 |
| same-minute median Spearman | +0.062754 | +0.097593 |
| positive same-minute groups | 93.10% | 96.55% |
| selected realized excess mean | +0.605437% | +0.512652% |
| selected day-balanced excess | +0.645989% | +0.522274% |
| selected bootstrap 95% lower | +0.057586% | -0.062522% |
| days with positive Spearman + selected excess | 4/5 | 2/5 |

Pooled improvement was +0.017472, narrowly below the preregistered +0.02
floor. Same-minute median improvement was +0.034839 and passed its +0.01 floor.

Daily market-regime ranking was positive on all five days, but the semantic
regression score >0 selection was economically unstable:
- Jun8 selected excess -0.0305%
- Jun9 -0.0034%
- Jun10 +1.8244%
- Jun11 -0.1241%
- Jun12 +0.9450%

## Interpretation

Market-wide momentum context is a real missing information axis for the
post-entry state: it improves rank ordering, especially after controlling for
minutes held.

The remaining bottleneck is converting the improved ranking into an absolute
economic HOLD/EXIT decision. The regression's zero boundary is not calibrated
robustly enough.

Next branch: keep rich-second + market-regime causal state fixed and directly
classify whether the fit-minute-baseline-adjusted multi-minute remaining value
is positive. This is a different target head, not another one-minute HOLD
classifier. Calibrate probability chronologically and use semantic P>0.5
without June threshold tuning.
