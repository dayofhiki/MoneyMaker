# Request 177 — equal-trading-day weighted hurdle EV result

Authoritative run: `36037073473`  
Artifact: `10825243130`

No new market dates were opened.

## Decision

**FAIL strict promotion gate, but day-level robustness nearly reached zero**

Compared with Request 176, equal-day weighting reduced the magnitude of the
negative bootstrap lower bound but did not cross zero.

Core June 8-12 results:
- EV Spearman: +0.11381
- EV-positive states: 1,172 / 5,662 = 20.70%
- selected realized excess mean: +1.2857%
- selected day-balanced excess: +1.2504%
- selected positive-target rate: 43.94%
- good days: 4/5
- trading-day bootstrap 95% interval: **[-0.0046%, +2.6290%]**

Daily selected realized excess:
- Jun 8: +0.1249%
- Jun 9: +0.0610%
- Jun 10: +3.4094%
- Jun 11: -0.1350%
- Jun 12: +2.7918%

The frozen gate failed only because the bootstrap lower bound remained slightly
negative.

## Interpretation

Request 176 and Request 177 are both economically strong but disagree on which
June day is weak. This indicates estimator sensitivity rather than a stable
single-model failure mode.

Do not select the better June model or tune either threshold.

The next candidate is a conservative cross-estimator consensus fixed before a
new evaluation block: require both the standard row-weighted hurdle EV and the
equal-day-weighted hurdle EV to be positive. Validate this candidate only on a
new unopened date block. June 8-12 is retired from model selection.
