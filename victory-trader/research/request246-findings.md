# Request246 findings — entry timing improves economics, gate still FAIL

Authoritative run: `36305423929`  
Artifact: `10927086532`  
Evaluated source: `e90a070dd0a6d32ee1475e1e786eb6008a73e3b0`.

No new market dates were opened. Request245 admission remained frozen at 0.60.
The downstream HOLD/EXIT teacher was the Request243 earlier teacher trained only
on Apr30-May4. Entry-value heads trained only on May5-8 and were evaluated on
already-open May11-20.

## Decision

**ECONOMIC GATE FAIL, BUT ENTRY-TIMING IMPROVEMENT IS REAL ON THIS DEVELOPMENT BLOCK.**

Training support:
- 412 admitted states;
- 322 finite ENTER targets across 93 episodes;
- 357 finite WAIT targets across 124 episodes;
- ENTER target mean -0.7656%, positive 28.57%;
- WAIT target mean -0.5211%, positive 17.37%.

The negative target means are themselves informative: even within causally
tradable states, most opportunities under the frozen downstream teacher were not
economically attractive.

## Test result

First-admitted baseline, same downstream teacher:
- 180 entries;
- 90.00% entry-fill coverage;
- 71.11% fully closed;
- 128 closed trades;
- BASE mean -1.4409%;
- STRESS mean -3.4205%;
- positive rate 23.44%;
- severe loss <= -2%: 36.72%.

Learned ENTER/WAIT:
- 51 entries;
- 98.04% entry-fill coverage;
- 86.27% fully closed;
- 44 closed trades;
- BASE mean -1.0264%;
- STRESS mean -3.0276%;
- positive rate 25.00%;
- severe loss <= -2%: 25.00%.

On the 44 episodes where both policies closed, learned timing improved BASE
return by +0.4561 percentage points on average. Both preregistered relative
improvement checks passed.

The gate failed only because:
- closed BASE mean remained negative;
- closed positive rate remained below 50%.

## Interpretation

Request245 established that execution availability is causally learnable.
Request246 now shows that ENTER versus WAIT timing is also learnable enough to
produce a measurable relative economic improvement without threshold tuning on
May11-20.

But entry timing alone cannot overcome the weak downstream position policy.
A large majority of executable admitted opportunities are still negative under
the frozen HOLD/EXIT teacher. The next isolated experiment should keep admission
and the learned entry policy fixed, then learn event-time HOLD versus EXIT from
causal position states using one-step continuation value under the same one-second
execution contract.

Do not retune the Request245 threshold or Request246 entry rule on May11-20.
If a position learner improves economics, a later experiment can perform one
joint policy-improvement iteration. If it does not, the remaining bottleneck is
the opportunity population/value representation rather than execution or entry
timing.
