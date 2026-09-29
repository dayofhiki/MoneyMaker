# Request293 — direct attractive-entry stopping

## Result

Request293 completed successfully through the consolidated research runner.
The research gate failed, but the experiment clarified the entry problem further.

## Direct current-entry value is learnable

Using the same 4,526 causal active-second states from Request292:

- OOF utility Spearman: 0.2247
- positive-utility rate: 59.70%
- positive-utility AUC: 0.5906

This is materially better than Request292's separate WAIT-minus-ENTER model,
whose one-step timing AUC was only 0.5095.

Therefore the current state's entry attractiveness is more learnable than a
separate forecast of future WAIT value.

## Policy behavior

The direct threshold policy entered the first active-second state whose predicted
current entry utility cleared a nested calibration threshold.

- episodes: 97
- entered: 81
- entry rate: 83.51%
- skipped: 16
- mean delay among entries: 16.7s
- median delay: 0s

This is much less procrastination than:
- Request291: 100% waited, ~193s mean delay
- Request292: 58.8% entered, ~84.5s mean delay among entries

The controller now behaves more like "enter if currently good enough" rather
than searching indefinitely for a perfect bottom.

## Entry timing economics

Among the 81 entered episodes:

- mean price improvement vs immediate: +0.103%
- chosen MFE: +5.428%
- immediate MFE on the same episodes: +5.414%
- MFE gain: +0.013pp

Tail opportunity rates were nearly preserved:
- +5%: 30.86% chosen vs 30.86% immediate
- +10%: 11.11% chosen vs 11.11% immediate
- +20%: 3.70% chosen vs 2.47% immediate

So the direct controller avoids the severe opportunity destruction seen in
Request291. It also beats the fixed-2%-pullback policy on coverage and
cash-adjusted utility.

## But immediate entry still wins on aggregate utility

Across all 97 episodes, with SKIP treated as cash:

- Request293 decision utility: +1.347%
- immediate-entry utility: +1.732%
- Request293 gap vs immediate: -0.385pp
- Request292 decision utility: +0.868%
- fixed 2% pullback cash-adjusted utility: +0.801%

Thus Request293 is a clear improvement over Request292 and the fixed 2% rule,
but it still discards or delays enough valuable setups that immediate entry
remains better under the current development utility.

## Interpretation

Three entry-control hypotheses have now been separated:

1. Fixed 2% pullback:
   useful low-price intuition but misses about half the selected setups.

2. Explicit WAIT-value prediction:
   not learnable enough; Request292 one-step AUC ~0.509.

3. Direct current-entry attractiveness:
   learnable; OOF Spearman ~0.225 and positive AUC ~0.591, with far healthier
   policy behavior.

The remaining timing gain is small because a scalar "current value above
threshold" does not identify the local turn. Median delay is zero, meaning many
trades are still immediate. When the policy waits, it does not explicitly know
whether price is still falling, stabilizing, or beginning to re-accelerate.

## Next boundary: Request294

Keep the crack setup selector frozen and keep direct entry value as the base
score. Do not restore a separate WAIT model.

Add a causal local-turn layer using only already-observed active seconds:

- falling pressure / recent negative-return streak;
- distance from active local low;
- bounce magnitude from that low;
- seconds since local low;
- short-window slope and acceleration;
- activity/volume re-expansion after the low;
- reclaim of recent micro-high;
- drawdown from active high.

The action should become:

- if direct entry value is poor: keep observing or SKIP;
- if value is good but price is still falling: do not catch the knife yet;
- if value is good and the local path shows stabilization/reversal:
  ENTER;
- if price re-accelerates without a deep pullback and waiting would mean
  chasing later: allow early ENTER.

Evaluate against Request293, immediate entry and fixed 2% pullback.

The key success criterion is not merely a cheaper price. Require:
- positive price improvement;
- no loss of MFE / +10% opportunity rate;
- higher cash-adjusted decision utility than immediate entry.

Do not begin HOLD/EXIT yet. The setup selector is promising, but the entry
controller has not yet beaten immediate entry.
