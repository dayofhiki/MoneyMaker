# Request 215 findings — event-time executable recurrent entry

## Verdict

Request215 **passes the structural replay objective but fails the economic gate**. It should not be promoted as a trading policy.

The experiment shows that Request214's exact-minute accounting was materially incomplete, but that incompleteness was not the reason the entry policy made zero trades. After event-time repair the policy is fully accounted and still abstains on all 223 episodes.

## What changed

Request214 required every exact decision minute and an exact +3 minute exit reference. Request215 keeps the feature family, hurdle model, timing model, BASE costs, five-minute entry window, EV>0 boundary and three-minute diagnostic target frozen while changing only replay semantics:

- a silent exact-minute decision point becomes a causal STALE/WAIT state;
- ENTER remains possible only at an observed executable checkpoint;
- a scheduled +3 minute diagnostic exit executes at the first available observation at or after the target;
- a pre-target observation can never be used as that exit;
- unresolved entered outcomes remain unresolved rather than cash.

## Canonical run

- Workflow run: `36167161063`
- Source commit: `984f333fac26e629ee1502b839b13830bf9370b5`
- Artifact: `10877699466`
- Python: 3.11.16
- Frozen numerical stack: same Request214 constraints
- Development dates: 2026-06-23, 24, 25, 26, 29
- New dates opened: **no**

## Replay integrity result

Across 223 episodes there are 1,115 five-minute clock checkpoints.

- observed checkpoints: **1,016**
- silent STALE checkpoints: **99 (8.88%)**
- episodes containing at least one stale wait: **55**
- episodes with no observed state at all: **0**

Request214 treated those 55 affected episodes as terminal coverage misses. Request215 carries the clock causally through silence and produces a complete decision for every episode.

Outcome coverage also improves materially:

- exact +3m evaluable watch states: **814 / 1,016 = 80.12%**
- event-time evaluable watch states: **930 / 1,016 = 91.54%**
- newly resolved states: **116**
- states with delayed exit: **116**
- mean execution delay beyond target: **0.438 min**
- p90 delay: **1.0 min**

So the previous exact-clock representation was throwing away a real amount of executable information.

## Economic result

The repaired policy still enters **zero** positions:

- ABSTAIN: **223 / 223**
- unresolved episodes: **0**
- positive predicted EV states: **0**
- predicted EV range: **-3.523% to -0.280%**
- mean predicted EV: **-1.428%**
- development gate: **FAIL**

This is stronger evidence than Request214. The zero-entry result is no longer confounded by 55 missing-clock episodes.

## Why the value model still refuses every trade

The repaired labels contain genuine profitable states:

- event-time evaluable states: **930**
- realized positive BASE states: **171**
- realized positive rate: **18.39%**

But the model barely separates them:

- positive-probability ROC AUC: **0.5939**
- mean predicted positive probability on actually positive states: **19.81%**
- mean on nonpositive states: **18.99%**
- expected-value Spearman: **+0.0646**

The magnitude heads imply a harsh payoff asymmetry:

- predicted positive magnitude mean: **+1.430%**
- predicted nonpositive magnitude mean: **-2.119%**
- mean break-even positive probability: **64.66%**
- maximum predicted positive probability: **28.06%**
- states above their own break-even probability: **0**

This is not a threshold problem. Lowering the EV>0 boundary would knowingly buy negative-expectation states.

The ranking audit confirms that point:

- top 10% by predicted EV: realized mean **-1.260%**, positive rate **18.28%**
- top 20% by predicted EV: realized mean **-1.283%**, positive rate **14.52%**

The current state representation does not isolate a high-quality pocket hidden just below the threshold.

## Relation to Request212

Request212 already showed that simply attaching the existing recurrent HOLD/EXIT model to Request208 entries did not rescue the economics:

- dynamic exit mean BASE: **-1.246%**
- dynamic exit equal-day mean: **-1.206%**
- fixed 3m Request208 equal-day mean: **-1.114%**
- HOLD advantage Spearman: **-0.0123**
- selected realized one-step HOLD advantage mean: **-0.0552%**
- dynamic minus fixed 3m equal-day difference: **-0.0846pp**

Therefore the next step should not tune an entry threshold or merely shorten/lengthen the fixed holding horizon.

## Next research boundary

Build an **event-time POSITION replay** on the same already-opened block.

1. Re-anchor positions to the entry selected by the causal entry controller.
2. Make decisions only when an executable observation exists.
3. Treat silent time as forced HOLD/no-action, not as a skipped state.
4. Define one-step HOLD value to the next actually executable observation.
5. Add only causal information known at the current event, including time since the previous observation; never use time-to-next-observation as a feature.
6. Compare the event-time HOLD/EXIT policy against Request212 and the frozen fixed-3m comparator.
7. If HOLD ranking still fails, stop trying to rescue the current representation by threshold tuning and move to a new state representation / policy-value objective.

No new holdout should be opened for this structural experiment.
