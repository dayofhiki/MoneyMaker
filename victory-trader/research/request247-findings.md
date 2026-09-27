# Request247 findings — HOLD/EXIT improves modestly, economic gate still FAIL

Authoritative run: `36305782241`  
Artifact: `10926884304`  
Evaluated source: `4737a6681ada0b93e2b6da2151cec4bc95522454`.

No new market dates were opened. Request245 admission and Request246 entry timing
were reconstructed unchanged. The only new component was a causal one-step
position-value regressor trained on May5-8.

## Decision

**ECONOMIC GATE FAIL. POSITION MANAGEMENT HELPS, BUT ONLY MODESTLY.**

Position-target support:
- 2,327 finite one-step HOLD targets across 230 episodes;
- 4,723 unresolved target rows;
- mean one-step HOLD advantage +0.0878%;
- positive one-step HOLD advantage 51.65%.

The large unresolved population remains an important limitation: position-state
execution is still much less complete than entry-state execution.

## Test result

Request246 learned entry + earlier HOLD/EXIT teacher:
- 51 entries;
- 98.04% entry-fill coverage;
- 86.27% fully closed;
- 44 closed trades;
- BASE mean -1.0264%;
- STRESS mean -3.0276%;
- positive rate 25.00%;
- severe loss <= -2%: 25.00%.

Request247 learned entry + learned one-step position policy:
- same 51 entries;
- 98.04% entry-fill coverage;
- 84.31% fully closed;
- 43 closed trades;
- BASE mean -0.9860%;
- STRESS mean -3.0134%;
- positive rate 34.88%;
- severe loss <= -2%: 20.93%.

On the 37 episodes where both policies closed, Request247 improved BASE return by
+0.1537 percentage points on average.

Relative-improvement checks passed, but the fixed economic gate failed because
the mean remained negative and the positive-trade rate remained below 50%.

## Interpretation

The research sequence now isolates three distinct facts:

1. Request245: tradability/execution availability is highly learnable and should
   remain a prerequisite layer.
2. Request246: ENTER versus WAIT timing is learnable enough to reduce losses
   materially relative to first-admitted trading.
3. Request247: HOLD versus EXIT value is also learnable enough to reduce losses
   and severe-loss frequency, but the incremental gain is smaller.

Even after all three layers improve in the expected direction, the realized
development economics remain negative. Therefore execution mechanics are no
longer the leading bottleneck, and simple controller refinement alone is unlikely
to be sufficient.

The next experiment should be diagnostic rather than another threshold tweak:
measure the positive-opportunity ceiling inside the causally tradable population
under the frozen Request247 downstream policy, then test whether current causal
features can rank those positive opportunities. If positive opportunities are
rare even for a diagnostic oracle, candidate population/admission is the
bottleneck. If they are common but poorly ranked, economic state representation
is the bottleneck.

Do not open June or a new holdout until that distinction is resolved.
