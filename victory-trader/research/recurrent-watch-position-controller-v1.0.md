# Request 205 — recurrent watch-entry + recurrent hold-exit controller v1.0

## Purpose

Request 204 validated the control structure "discover -> watch -> sometimes
wait -> sometimes enter" but showed that entry timing alone cannot overcome the
frozen mechanical post-entry rule. Even the hindsight-best minute-1..10 entry
under that fixed exit rule remained negative.

Request 205 keeps the recurrent flat-state structure and replaces the fixed
+10%-partial / trailing profit-taking rule with the intended continuous
post-entry behavior:

- if the remaining opportunity still appears better than exiting now: HOLD;
- otherwise: EXIT;
- an independent -3% hard stop remains active continuously as downside
  protection.

This is the first development experiment where a candidate can be watched
before entry and then repeatedly reconsidered after entry.

## Data split

No new dates.

- model training: Request-171 FIT only;
- evaluation: Request-171 chronological CALIBRATION only;
- Request-178 remains sealed.

## Entry controller

Use the simpler BASE_FEATURE_CONTROLLER from Request 204.

Reason fixed before Request 205 evaluation:
- Request 204 showed the recurrent action structure was useful;
- the ten extra watch-path features did not improve economics over the simpler
  controller and are therefore retired;
- no Request-204 threshold is tuned.

Flat-state action semantics remain exactly:

- ENTER_NOW when predicted enter-now value is positive and no worse than
  predicted wait value;
- WAIT when waiting is predicted better;
- DROP when both are non-positive;
- unavailable entry fills return to flat observation;
- final minute-10 checkpoint has no WAIT action.

Entry execution remains causal one-second BASE execution with 1-second latency
and 5-second expiry.

## Post-entry value model

Train one equal-day HistGradientBoostingRegressor on Request-171 FIT position
states only.

Target:
- best future BASE net exit return still attainable after the current state.

Inputs:
- the existing Request-191 causal position feature family only.

No Request-171 CALIBRATION outcome is used to fit an offset, threshold, feature
subset, or model parameter.

Model:
- squared error;
- learning_rate=0.05;
- max_iter=180;
- max_leaf_nodes=15;
- min_samples_leaf=75;
- l2_regularization=2.0.

## HOLD / EXIT action

At each completed one-minute position state:

1. compute a causal current mark using the current completed close, BASE sell
   friction, and the actual modeled one-second entry price;
2. predict the best future BASE return still available;
3. HOLD only when predicted future best return is greater than the causal
   current marked return;
4. otherwise submit EXIT.

No +5%, +10%, capture percentile, trailing-profit threshold or calibrated
probability gate is used.

This directly encodes:
"if it still looks better to stay in than to cash out now, keep it."

## Independent hard stop

The HOLD model may never override the hard stop.

- stop reference: -3% from modeled entry price;
- evaluated on completed one-second OHLC;
- stop trigger becomes actionable only after that second is complete;
- first later eligible one-second open after 1-second latency is the exit fill;
- a halt/gap can therefore lose more than 3%;
- missing seconds never fabricate an exit.

A discretionary EXIT uses the same first-later-eligible-open rule.

Research cap remains 30 minutes from entry solely as a safety cap.

## Evaluation comparisons

On the same Request-204 recurrently selected calibration entries report:

1. Request-204 frozen runner-rule outcome;
2. Request-205 recurrent HOLD/EXIT outcome;
3. non-executable best-exit oracle for the same admitted entries.

Report:
- watch episodes;
- admitted entries;
- position-start coverage;
- completed coverage;
- mean and day-balanced return;
- positive rate;
- average winner / loser;
- payoff ratio;
- severe-loss rate;
- positive days;
- average holding time;
- hard-stop exits;
- discretionary exits;
- forced-cap exits;
- day-balanced bootstrap interval;
- matched improvement versus the frozen runner rule.

Also report post-entry value-model ranking correlation on calibration position
states as a diagnostic, but do not use it to tune a gate.

## Development success rule

Request 205 passes only if:

1. at least 30 admitted entries;
2. position-state construction covers >= 90% of admitted entries;
3. completed outcome coverage >= 90%;
4. day-balanced BASE return > 0;
5. at least 6 of 8 days are positive;
6. day-balanced bootstrap 95% lower bound > 0;
7. matched day-balanced improvement versus Request-204 frozen runner exit is at
   least +0.75 percentage points;
8. severe-loss rate is no worse than the frozen runner rule.

Passing remains development-only and does not automatically open Request-178.

## Failure rule

If Request 205 fails, do not tune the -3% stop, model threshold, holding cap,
minute cadence or regression hyperparameters on calibration outcomes.

Use the diagnostics to decide whether:
- post-entry value prediction itself is weak;
- the one-minute decision cadence is too coarse;
- the recurrent entry target must be retrained using dynamic-exit outcomes;
- or the candidate population remains too weak.
