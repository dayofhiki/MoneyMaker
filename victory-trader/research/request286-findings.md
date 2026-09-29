# Request286 — completed-second context at original pullback

## Result

Request286 completed successfully and failed the overall second-information
gate, but it exposed one useful asymmetry.

At the exact original first causal 2% pullback decision time, adding only
completed one-second aggregates from the trailing 60 seconds did not improve
general value or winner prediction. It did materially improve severe-loss
classification.

No future second, delayed entry, new trading date, or execution change was
introduced.

## Frozen support and economics

- stage-1-shortlisted resolved pullback trades: 36
- actual trade mean: -0.64794%
- positive rate: 30.56%
- severe-loss rate <= -2%: 36.11%
- original entry and downstream outcome were identical in both models

## Baseline versus second-enriched

Baseline, 122 features:
- value Spearman: 0.1609
- positive AUC: 0.4800
- severe-loss AUC: 0.5619

Completed-second enriched, 159 features:
- value Spearman: 0.0649
- positive AUC: 0.3782
- severe-loss AUC: 0.6589

Change:
- value Spearman: -0.0960
- positive AUC: -0.1018
- severe-loss AUC: +0.0970

Only one of the three prediction metrics improved by at least 0.05, so the
preregistered information gate failed.

## Timing and coverage

The second features consume only completed one-second aggregates from the
trailing 60 seconds ending at decision_t - 1 second.

- all 330 pullback states had some second data
- 330 ticker-days were fetched
- no second after the decision timestamp was used
- entry time and downstream outcome were unchanged

Feature-level coverage varies because some short-window features require
enough observed activity, but the research population had full ticker-level
second-data coverage.

## Interpretation

The missing first-pullback information is asymmetric.

Completed sub-minute price/volume structure carries meaningful information
about imminent downside risk inside the stage-1 shortlist, where the
minute-level risk model previously collapsed:

- severe AUC improved from about 0.562 to 0.659.

However, the same second-resolution representation made winner/value ranking
worse:

- positive AUC fell from 0.480 to 0.378;
- value Spearman fell from 0.161 to 0.065.

Therefore the next model should not treat one-second context as a generic
"better state representation." Its current useful role is specifically as a
downside-risk sensor.

The support is still small at 36 resolved shortlisted trades, so this is a
development signal, not evidence of a stable profitable policy.

## Combined lesson from Requests284-286

Request284:
- adapting the first-pullback risk model to the shortlist did not repair the
  signal.

Request285:
- waiting one full observed minute improved some prediction metrics but made
  entry economics dramatically worse.

Request286:
- looking inside the final pre-decision minute at one-second resolution
  improves severe-loss discrimination without paying any entry delay, but does
  not identify winners.

This suggests that the relevant information arrives on a sub-minute timescale,
but the model should use it for risk control rather than winner selection.

## Next boundary

Test fixed causal waits of 5, 15 and 30 seconds after the first 2% pullback.

For each fixed horizon:
- never select the best future wait duration per episode;
- use only seconds completed through that fixed timestamp;
- re-anchor the actual entry price at that timestamp;
- preserve the same downstream stop/trailing/max-hold contract;
- compare how much severe-risk discrimination improves versus how much entry
  price/economics are sacrificed.

The purpose is to locate the information-versus-delay frontier before building
a recurrent ENTER / WAIT / REJECT controller.

Do not open May11-20 or June15-19 yet.
