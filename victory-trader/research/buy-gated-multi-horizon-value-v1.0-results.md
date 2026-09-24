# Request 169 — direct multi-horizon continuation-value result

Authoritative run: `36031056397`  
Artifact: `10822225777`

No new market dates were opened.

## Decision

**FAIL**

All three exact medium-horizon value models generalized in the wrong direction
on June 8-12.

| horizon | target coverage | Spearman | predicted-positive realized mean | day-balanced mean |
|---|---:|---:|---:|---:|
| 2m | 75.86% | -0.0417 | -0.0213% | -0.0297% |
| 5m | 66.24% | -0.0421 | -0.0721% | -0.0527% |
| 10m | 52.18% | -0.0717 | -0.1168% | -0.0837% |

The preregistered chosen-horizon diagnostic also failed:
- evaluable coverage 62.66% vs >=70% gate;
- Spearman -0.0701 vs >=+0.05 gate;
- selected realized advantage -0.0878%;
- selected day-balanced advantage -0.0698%;
- bootstrap 95% lower -0.3799%;
- both-positive days 2/5.

## Diagnosis

The current causal bar/second/path state does not robustly predict the exact
timing of a better exit, even at 2/5/10-minute lookaheads.

Together with Requests 166-168, this rejects further reshuffling of the same
minute/second/path feature family for HOLD/EXIT timing.

The next branch should test a genuinely new causal intraday information source.
Historical NBBO is unavailable under the configured entitlement (prior
REST/Flat File probes both returned 403), so the next feasible source to audit
is historical tick-level trades already supported by MassiveClient.trades().
