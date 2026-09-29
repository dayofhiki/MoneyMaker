# Request284 — shortlist-adapted stage-2 severe-risk findings

## Result

Request284 completed successfully and failed the shortlist-adapted risk gate.

Request283 showed that the global severe-risk score loses most of its ordering
inside the stage-1 shortlist. Request284 therefore changed only the risk
learner, not the trading action.

The adapted model:
- added causal stage-1 predicted value and margin to stage-2 context;
- gave nested-cross-fitted shortlist-like training pullbacks 4x weight;
- excluded the outer held-out day from every stage-2 fit;
- generated each outer-training shortlist-emphasis flag without that row's own
  downstream target.

## Conditioned comparison

Resolved stage-1-shortlisted pullback trades: 36

Global Request283-style risk model:
- AUC: 0.5619
- average precision: 0.4178
- AP lift over severe prevalence: +0.0567
- rejected trade mean: -0.8892%
- accepted trade mean: -0.4067%
- rejected severe rate: 44.44%
- accepted severe rate: 27.78%
- severe-rate gap: +16.67pp

Shortlist-adapted model:
- AUC: 0.5418
- average precision: 0.4039
- AP lift over severe prevalence: +0.0428
- rejected trade mean: -0.4433%
- accepted trade mean: -0.8311%
- rejected severe rate: 41.18%
- accepted severe rate: 31.58%
- severe-rate gap: +9.60pp

AUC gain versus global: -0.0201

Every preregistered signal check except support failed.

## Interpretation

The conditioning problem is not repaired by telling the risk model that a
candidate was strong at HOT time or by reweighting training toward shortlist
examples.

At the first 2% pullback, the current causal representation simply does not
contain enough stable information to order outcomes inside the stage-1
shortlist. Additional fitting pressure makes the model over-specialize without
creating signal.

This is important because it argues against threshold tuning and against
immediately repeating the same weak risk decision recurrently.

## Next boundary — Request285

Test whether the problem becomes more separable after new causal information
arrives.

For each stage-1-shortlisted candidate that reaches the first 2% pullback,
construct fixed recheck states at:
- trigger state + 0 observed events;
- +1 event;
- +2 events;
- +3 events.

At each fixed horizon:
- use only the price path and completed-minute information available through
  that state;
- treat that state as a fixed entry, never choose the best later state;
- apply the same frozen -2.5% stop / 2% trailing / 10-minute max-hold exit;
- measure raw delayed-entry economics and leave-one-day-out learnability.

The question is not yet whether a WAIT policy is profitable. It is whether
waiting for one or more new states turns an unlearnable first-pullback problem
into a learnable one.

Do not open May11-20 or June15-19.
