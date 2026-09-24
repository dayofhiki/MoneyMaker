# Request 166 — BUY-gated POSITION observability results

Authoritative run: `36027818439`  
Artifact: `10821086716`

No new market dates were opened.

## Result

**FAIL**

Request 165's entry gate improved the population, but the unchanged Request-145
POSITION controller did not generalize to the new BUY population.

Evaluation June 8-12:
- BUY anchors: 286
- anchors with a causal post-entry path: 256
- anchor-to-path coverage: **89.5105%**
- position decision rows: 6,230
- within-path executable decision-state coverage: **91.3002%**
- HOLD AUC: **0.495862**
- semantic HOLD rows at calibrated P>0.5: **1**
- remaining-option global Spearman: **0.089999**
- same-held-minute Spearman median: **0.053274**
- positive same-minute groups: **93.10%**
- predicted-excess-positive realized excess mean: **+0.5120%**
- day-balanced realized excess: **+0.5782%**
- all-sign-positive days: **1/5**

The one-minute HOLD classifier has collapsed to essentially random ordering on
the stricter BUY population. In contrast, residual multi-minute option value
remains weakly but consistently observable.

## Diagnosis

The validated entry gate does not solve POSITION timing by itself. The current
Request-145 state mainly describes the present level/path summary; it lacks the
exact multi-lag path-transition family that previously stabilized continuation
ordering in expanded-history v3.7.

Next: add only the previously supported exact 1/2/3/5/8/13-minute transition
deltas of causal position-path state, keep target/model/action boundary frozen,
and test whether the short-horizon HOLD signal returns. This is a no-new-date
controller diagnostic, not a recurrent-profit claim.
