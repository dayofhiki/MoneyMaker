# Request 171 — rich one-second POSITION value result

Authoritative run: `36033120106`  
Artifact: `10823097946`

No new market dates were opened.

## Decision

**FAIL strict promotion gate, but materially more robust selected-set economics**

Historical tick trades/NBBO were not used. Rich causal one-second aggregate
features were available on 100% of evaluation states.

| metric | Request-166 baseline | rich-second candidate |
|---|---:|---:|
| pooled Spearman | +0.089999 | +0.095393 |
| same-minute median Spearman | +0.053274 | +0.062754 |
| positive same-minute groups | 93.10% | 93.10% |
| selected realized excess mean | +0.512031% | +0.605437% |
| selected day-balanced excess | +0.578221% | +0.645989% |
| selected bootstrap 95% lower | -0.068508% | **+0.057586%** |
| days with positive Spearman + selected excess | 4/5 | 4/5 |

The candidate's pooled Spearman improvement was only +0.005394 versus the
predeclared +0.03 requirement. Same-minute median improvement was +0.009480
versus +0.02 required. Therefore promotion_gate_pass=false.

## Daily candidate behavior

- Jun 8: Spearman +0.0550, selected excess +0.1655%
- Jun 9: Spearman +0.0683, selected excess +0.3706%
- Jun10: Spearman +0.0815, selected excess +1.4635%
- Jun11: Spearman +0.0511, selected excess -0.2069%
- Jun12: Spearman +0.2916, selected excess +1.4372%

## Interpretation

Unlike Requests 167-169, the richer one-second information did not reverse or
destroy the remaining-value signal. It modestly improved ranking and converted
the selected-set day-cluster bootstrap lower bound from negative to positive.

The strict material-improvement gate still fails, so the full rich-second model
must not be promoted as validated.

The failure pattern suggests a calibration/uncertainty problem more than a
complete absence of value information: the score is positively ranked on every
evaluation day, but a positive point estimate still selects a negative realized
set on Jun11.

Next test: keep the rich-second feature model fixed and build a conservative
fit/calibration-only lower-confidence value bound. Test whether only states
whose conservative lower bound is above zero form a stable patience set before
constructing a recurrent controller. No threshold may be tuned on June 8-12.
