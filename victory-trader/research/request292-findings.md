# Request292 — fixed-horizon fitted stopping

## Result

Request292 completed successfully through the consolidated research runner.
The overall gate failed, but it resolved Request291's chronic-WAIT pathology and
isolated the next bottleneck.

## Setup sanity check at matched cardinality

The Request291 rich setup selector was not clearly superior to the 38-feature
baseline when both were forced to select the same 105 candidates:

- baseline day-balanced mean MFE: +4.67%
- rich day-balanced mean MFE: +4.52%
- +5% rate: 25.60% baseline vs 27.29% rich
- +10% rate: 8.53% baseline vs 8.35% rich
- +20% rate: 3.92% baseline vs 2.61% rich

Therefore the crack-selection architecture remains useful, but the extra
multi-minute feature block is not yet proven better than the simpler base
representation. Do not attribute the R291 tail enrichment to the extra feature
block alone.

## Fixed terminal and event state

Request292 evaluated every entry candidate against the same HOT+60m terminal.

- selected setup candidates: 105
- complete active-second episodes: 97
- active-second states: 4,526
- second-feature coverage: 100%
- fixed 2% pullback available: 53 candidates

The event-driven second-resolution state remains viable.

## Fitted WAIT signal

The fitted continuation model did not learn the next-state timing advantage:

- one-step WAIT beneficial rate: 41.39%
- predicted WAIT-minus-ENTER Spearman: 0.0248
- AUC: 0.5095

This is effectively random for the direct next-state timing question.

## Policy behavior

The fitted ENTER/WAIT/SKIP policy:

- entered: 57 / 97 = 58.76%
- skipped: 40
- mean delay among entries: 84.5s
- median delay: 34s

This is much healthier than Request291's 100% WAIT / ~193s mean delay.

Among the 57 entries:
- mean entry-price improvement vs immediate: +0.559%
- chosen MFE: +4.70%
- immediate-entry MFE on the same episodes: +4.38%
- MFE gain: +0.319pp
- +10% opportunity rate: 10.53% chosen vs 8.77% immediate
- +20% opportunity rate: 1.75% chosen vs 0% immediate

So event-driven waiting can improve the actual price and preserve/increase
upside on the subset that the policy ultimately enters.

## But the total decision policy is worse than immediate

Across all 97 episodes, treating SKIP as cash:

- Request292 mean decision utility: +0.868%
- immediate-entry mean utility: +1.732%
- loss versus immediate: -0.865pp

The policy discarded too many valuable setups. The SKIP/WAIT decision boundary,
not the existence of useful lower entry prices, is now the principal entry
bottleneck.

The policy did narrowly beat the fixed-2%-pullback baseline on cash-adjusted
utility:
- Request292: +0.868%
- fixed 2% pullback: +0.801%

The 2% baseline still only covered 53/97 episodes.

## Interpretation

Request292 gives two simultaneous facts:

1. timing opportunity is real enough that selected entries got ~0.56% cheaper
   and retained more MFE;
2. a separate learned WAIT-value model is not the right control primitive here.

The continuation Q estimate is nearly random, while the policy's useful behavior
appears to come from recognizing attractive current entry states and from
calibrated abstention.

## Request293 boundary

Keep the Request291 setup population frozen and simplify the controller.

Train only a direct causal model of:
"How attractive is ENTER at the next executable active-second price, under the
shared HOT+60m terminal and the same upside-minus-adverse-excursion utility?"

At every new active second:
- if current predicted ENTER value clears a train/calibration-derived threshold,
  ENTER immediately;
- otherwise keep observing;
- if no state clears the threshold in the observation window, SKIP.

Do not separately predict WAIT. WAIT is simply the consequence of the current
state not yet being attractive enough.

Evaluate:
- direct-value OOF Spearman / positive AUC;
- entry rate and delay;
- price improvement vs immediate;
- MFE and MAE vs immediate on the same entered episodes;
- cash-adjusted utility vs immediate and fixed 2% pullback.

This is closer to the intended trader behavior: do not predict a magical future
bottom; repeatedly ask whether the currently offered price is good enough.
