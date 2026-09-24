# Request 171 — rich one-second POSITION value result

Authoritative run: `36033120106`  
Artifact: `10823097946`

No new market dates were opened.

## Decision

**NO PROMOTION, but meaningful partial success**

The 20 preregistered rich one-second aggregate features were available on
100.0% of evaluation POSITION states and improved the surviving
remaining-option signal, but not by the material-improvement margins frozen
before evaluation.

| metric | Request-166 baseline | rich-second candidate | change |
|---|---:|---:|---:|
| pooled Spearman | +0.0900 | +0.0954 | +0.0054 |
| same-minute median Spearman | +0.0533 | +0.0628 | +0.0095 |
| positive same-minute groups | 93.10% | 93.10% | 0 |
| predicted-positive rate | 28.26% | 27.71% | -0.55pp |
| selected realized excess mean | +0.5120% | +0.6054% | +0.0934pp |
| selected day-balanced excess | +0.5782% | +0.6460% | +0.0678pp |
| cluster-bootstrap 95% lower | -0.0685% | **+0.0576%** | crossed above zero |
| positive Spearman + selected excess days | 4/5 | 4/5 | unchanged |

The candidate's own observability bridge passes, because:
- pooled and same-minute ranking are positive;
- selected realized value is positive;
- 4/5 days have both positive ranking and selected realized excess;
- most importantly, the five-day cluster-bootstrap lower bound is now positive.

However the preregistered **promotion** gate fails because the incremental
ranking improvements were smaller than required:
- pooled improvement +0.0054 < +0.03 required;
- same-minute median improvement +0.0095 < +0.02 required.

Daily candidate metrics:
- Jun 8: Spearman +0.0550, selected excess +0.1655%
- Jun 9: Spearman +0.0683, selected excess +0.3706%
- Jun 10: Spearman +0.0815, selected excess +1.4635%
- Jun 11: Spearman +0.0511, selected excess -0.2069%
- Jun 12: Spearman +0.2916, selected excess +1.4372%

## Interpretation

Request 171 is the first post-entry information intervention in the current
Request-165 BUY distribution to improve both pooled ranking and selected
economic value without worsening the signal family. It is not strong enough to
promote as the final controller, but it demonstrates that one-second
microstructure-like information contains incremental post-entry information.

Do not loosen the Request-171 improvement thresholds or select individual rich
features based on June 8-12.

Next research should keep June 8-12 closed for tuning and use fit/calibration
history only to determine whether a smaller, development-selected
microstructure representation or regime-conditioned model can materially
strengthen the signal. The June block may then be used only as a fixed
diagnostic comparison, not as a selection source. If a development-only
candidate is frozen, later unopened dates are required for promotion to a
recurrent HOLD/EXIT controller.
