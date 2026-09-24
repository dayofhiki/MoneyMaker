# Request 176 — market-regime hurdle expected remaining-value result

Authoritative run: `36036428387`  
Artifact: `10824792636`

No new market dates were opened.

## Decision

**FAIL strict promotion gate, but strongest post-entry result so far**

The three-head hurdle expected-value formulation substantially improved the
economic semantics of the post-entry signal without a June-tuned probability
threshold.

Core results on June 8-12:
- EV Spearman: **+0.125001**
- EV-positive states: **1,213 / 5,662 = 21.42%**
- selected realized excess mean: **+1.3712%**
- selected day-balanced excess mean: **+1.2545%**
- selected positive-target rate: 45.42%
- good days with both positive ranking and positive selected excess: **4/5**
- trading-day bootstrap 95% interval: **[-0.0357%, +2.7960%]**

The frozen gate failed only because the bootstrap lower bound remained slightly
below zero.

Daily selected realized excess:
- Jun 8: +0.1358%
- Jun 9: -0.1706%
- Jun 10: +3.8365%
- Jun 11: +0.0134%
- Jun 12: +2.4573%

Daily EV ranking was positive on all five days.

## Interpretation

Request 175 showed that P(value>0)>0.5 is the wrong economic boundary because
probability calibration compresses around a roughly one-third base rate while
positive outcomes are much larger in magnitude than non-positive outcomes.

Request 176 resolves much of that mismatch by combining:
- calibrated probability of positive remaining value;
- conditional positive magnitude;
- conditional non-positive magnitude.

The semantic EV>0 boundary produces materially broader and more economically
useful support than the Request-174 P>0.5 rule.

Do not relax the bootstrap gate after seeing June.

The main remaining weakness is day-level instability: a small number of strong
days contribute disproportionately to the selected average. The next causal
development intervention should therefore target day robustness rather than a
new June threshold. A principled next candidate is equal-trading-day weighting
during probability and magnitude learning/calibration, with the same features,
target and EV>0 boundary.
