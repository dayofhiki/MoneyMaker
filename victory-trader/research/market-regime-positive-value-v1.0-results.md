# Request 174 — market-regime positive remaining-value classifier result

Authoritative run: `36034915398`  
Artifact: `10824555305`

No new market dates were opened.

## Decision

**FAIL**

The direct classifier retained weak same-minute ordering information but its
chronologically calibrated semantic P>0.5 boundary collapsed support.

- market context coverage: 100%
- global AUC: 0.545151 vs 0.55 floor
- median same-minute AUC: 0.536828 vs 0.53 floor
- positive same-minute groups: 79.31%
- unconditional positive-target rate: 32.78%
- P>0.5 selected rows: 9 / 5,662 = 0.159%
- selected positive-target rate: 44.44%
- selected realized excess mean: +0.3102%
- selected day-balanced excess: +1.2926%
- bootstrap 95% interval: [-1.8301%, +4.2400%]
- good days: 2/5

Daily AUC:
- Jun8: 0.5575, 0 selected
- Jun9: 0.4874, 4 selected, all target-negative
- Jun10: 0.5631, 4 selected, 75% target-positive
- Jun11: 0.5090, 1 selected, target-positive
- Jun12: 0.6604, 0 selected

## Interpretation

Market-regime + rich-second state carries some ranking information but a
Platt-calibrated absolute probability boundary is not stable enough for a
runtime decision. Do not tune the 0.5 threshold on June.

The next step is a frozen distribution/calibration audit across fit,
calibration and June evaluation: positive-target prevalence, raw-score and
calibrated-probability distributions, score bins, minutes-held composition,
and day-level drift. If the ranking survives while prevalence/intercept shifts,
the next policy head should model economic expected value or regime-conditioned
base rate rather than another fixed probability threshold.
