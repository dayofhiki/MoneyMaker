# Request 169 — direct multi-horizon continuation-value result

Authoritative run: `36031056397`  
Artifact: `10822225777`

No new market dates were opened.

## Decision

**FAIL**

The direct 2m/5m/10m continuation-value formulation did not generalize on the
June 8-12 BUY-gated POSITION population.

| horizon | target coverage | Spearman | predicted-positive realized mean | day-balanced mean |
|---|---:|---:|---:|---:|
| 2m | 75.86% | -0.0417 | -0.0213% | -0.0297% |
| 5m | 66.24% | -0.0421 | -0.0721% | -0.0527% |
| 10m | 52.18% | -0.0717 | -0.1168% | -0.0837% |

The preregistered chosen-horizon diagnostic also failed:
- evaluable coverage: 62.66% versus 70% minimum;
- Spearman: -0.0701 versus +0.05 minimum;
- selected rate: 58.53%;
- selected realized advantage: -0.0878%;
- selected day-balanced advantage: -0.0698%;
- trading-day bootstrap 95% interval: [-0.3799%, +0.2403%];
- both daily ranking and selected value positive on 2/5 days versus 4/5 minimum.

The score chose 10m most often (3,230 of 6,230 rows), then 2m (2,079), then
5m (921), despite 10m having the weakest coverage and most negative fresh
ranking. This is consistent with poor out-of-period value calibration rather
than a useful adaptive horizon signal.

## Interpretation

Request 169 rejects a simple direct exact-horizon regression family on the
current Request-166 causal POSITION state. It also reinforces the Request
166-168 evidence that current minute-level POSITION information does not yet
support a robust executable stopping controller on the new Request-165 BUY
distribution.

Do not tune horizon weights, choose 2m post hoc, or invert the negative June
scores. The June block is already opened.

The next branch should diagnose why the surviving oracle-style remaining-option
signal is much weaker on this BUY distribution and why direct continuation
targets reverse out of period. Priority should be on genuinely new causal
intraday/execution information or target conditioning, not more thresholds,
path-transition copies, or horizon selection.
