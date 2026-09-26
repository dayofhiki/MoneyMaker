# Request247 findings — one-event redecision creates WAIT support and a small positive BASE subset, but gate FAILS

Authoritative run: `36262004252`  
Artifact: `10912147665`  
Repository tests and Flat Files checks passed.

No new dates were opened. No class balancing or May-tuned threshold was used.

## Verdict

Request247 **fails the preregistered gate**, but it is the strongest structural improvement in the recurrent-entry branch so far.

The fixed one-observed-event bootstrap repairs the WAIT-label starvation seen in Request246:

| WAIT training support | Request246 | Request247 |
|---|---:|---:|
| conditional training rows | 788 | 676 |
| positive rows | 3 | **137** |
| positive rate | 0.38% | **20.27%** |

The median gap to the next eligible observed state is one minute.

## Calibration behavior

| Metric | Request243 | Request246 | Request247 |
|---|---:|---:|---:|
| Entries | 173 | 114 | **33** |
| Resolved | 100 | 75 | **19** |
| Unresolved | 73 | 39 | **14** |
| Resolved BASE mean | -1.6464% | -0.8747% | **+0.5236%** |
| STRESS mean | -3.6270% | -2.9492% | **-1.4968%** |
| Severe loss <= -2% | 48.0% | 34.67% | **15.79%** |
| Initial WAIT actions | 442 | 1 | **103** |
| HOLD actions | 1,437 | 511 | **114** |

The positive BASE mean is based on only 19 resolved trades. Fourteen selected trades remain unresolved, every daily aggregate remains unknown, and STRESS economics remain negative. There is **no valid portfolio return, compounded balance or promotion claim**.

The formal gate fails because:
- resolved trades are **19**, below the preregistered 20;
- chosen ENTER advantage precision is **52.63%**, below 55%.

## Decision quality

On comparable reachable states:

- flat best-action accuracy: **65.62%**
- chosen ENTER rows: **19**
- chosen ENTER truly beat WAIT and cash: **52.63%**
- chosen ENTER mean advantage over best alternative: **+0.5236%**
- chosen WAIT rows: **52**
- chosen WAIT truly beat ENTER and cash under the *actual recurrent policy*: **1.92%**
- chosen WAIT mean realized advantage: **-0.6457%**
- chosen HOLD rows: **91**
- chosen HOLD positive realized advantage: **72.53%**
- chosen HOLD mean advantage: **+1.1061%**

Direct HOLD classification continues to improve and now looks materially useful on this development diagnostic.

## Interpretation

Request247 proves that the previous WAIT starvation was caused by the teacher target, not by an inherent absence of one-step patience opportunities. Replacing learned-teacher continuation with exactly-one-next-event entry increases positive WAIT training support by more than fifty-fold.

It also sharply reduces overtrading and severe losses, and the small resolved BASE subset becomes positive.

However, a new mismatch is exposed:

- training says WAIT means **skip this event, then enter at the next eligible event**;
- inference says WAIT means **skip this event, then re-run the policy**, which may WAIT again or ABSTAIN.

Therefore the learned one-event WAIT signal can be useful while the realized recurrent WAIT continuation is not. The very low 1.92% realized chosen-WAIT advantage is consistent with this target-policy mismatch.

## Next research boundary

Before opening any fresh holdout, Request248 should audit the WAIT transition itself on the same May block:

1. for every decision-reachable chosen WAIT, inspect exactly the next observed eligible state;
2. measure the forced-entry value at that next state under the same frozen HOLD/EXIT policy;
3. compare it with the actual recurrent-policy WAIT continuation;
4. record whether the next policy action is ENTER, WAIT or ABSTAIN;
5. quantify how much one-event opportunity is lost because the redecision policy does not follow through.

If the next-event forced-entry value is genuinely positive but the recurrent continuation destroys it, the next model should make the WAIT target and rollout semantics self-consistent. If the next-event value itself does not transfer, the one-event classifier is simply not generalizing and should not be promoted.

Request247 is promising evidence, not a profitability result.
