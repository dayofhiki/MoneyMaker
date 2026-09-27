# Request245 findings — tradability admission PASS, economics still negative

Authoritative run: `36304786758`  
Artifact: `10926827811`  
Evaluated source: `df8b6327b3098e659fdf48e6aa21db47168b349a`.

No new market dates were opened. May11-20 was already opened by Request243 and is
development/calibration evidence only.

## Decision

**STRUCTURAL PASS. ECONOMIC PASS NOT CLAIMED.**

The causal tradability head was trained on Apr30-May4. The admission threshold
was selected only on May5-8 using the preregistered lowest-threshold rule, then
frozen at 0.60 before May11-20 was scored.

On May11-20:
- 1,916 eligible WATCH states;
- tradable base rate 38.05%;
- ROC AUC 0.91499;
- average precision 0.88877;
- 643 states admitted;
- admitted tradability precision 83.20%, recall 73.39%;
- admitted immediate-entry availability 90.36%;
- admitted continuation availability 90.36%.

The frozen Request243 controller under the same one-second execution contract:
- 172 entries;
- entry-fill coverage 37.79%;
- fully closed resolution 17.44%;
- 30 closed trades;
- closed BASE mean -0.9795%.

With the causal tradability gate:
- 29 entries;
- 265 ENTER actions blocked and converted to WAIT;
- entry-fill coverage 89.66%;
- fully closed resolution 65.52%;
- 19 closed trades;
- closed BASE mean -1.3155%;
- closed STRESS mean -3.2824%;
- positive closed trades 31.58%;
- severe-loss rate <= -2%: 47.37%.

Every preregistered structural gate passed. Execution observability is therefore
a learnable causal state property with the current features.

## Interpretation

Request244's execution failure was not an unavoidable data mystery. The model can
identify, before entering, moments where prompt execution and near-term continued
activity are likely. This is a real architectural improvement and should remain
in the trader.

But filtering for tradability did **not** create positive economics. In fact the
closed-trade BASE mean worsened from -0.98% to -1.32%. This means liquidity/
tradability and economic opportunity are separate dimensions. A liquid trade can
still be a bad trade.

Do not tune the 0.60 gate on May11-20 and do not weaken it to recover trade count.
The next experiment should keep the gate frozen and relearn ENTER versus WAIT
inside the admitted population. The downstream policy used to create value labels
must itself be frozen and chronological; no future-best-state target or percentile
search on May11-20 is allowed.

The eventual system should treat tradability as a prerequisite, not as evidence
of expected profit.
