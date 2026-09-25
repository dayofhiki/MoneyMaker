# Request 217 findings — pullback phase ranking

## Verdict

Request217 **fails**. Explicit pullback-phase interactions add only a tiny amount of ranking information and do not create a profitable recurrent entry policy.

The result closes another branch of the diagnosis: the failure is not fixed by (a) exact event-time replay, (b) a direct cost-cover classifier, or (c) hand-encoding falling/stabilizing/turning/re-acceleration phase from the existing feature family.

## Canonical run

- Workflow run: `36168999495`
- Artifact: `10879651681`
- Development dates: 2026-06-23, 24, 25, 26, 29
- New dates opened: **no**
- Promotion eligible: **no**

## Same-capacity baseline

State-level cost-cover AUC: **0.58530**

First recurrent calibration-threshold crossing:

- attempts: **80**
- resolved: **72**
- resolved coverage: **90.0%**
- mean BASE: **-1.51116%**
- day-balanced BASE: **-1.51141%**
- positive rate: **16.67%**
- positive days: **0 / 5**

## Phase-aware model

State-level cost-cover AUC: **0.59276**

AUC uplift over same-capacity baseline: **+0.00746**, far below the preregistered +0.03 requirement and still below the 0.63 absolute gate.

First recurrent calibration-threshold crossing:

- attempts: **67**
- resolved: **59**
- resolved coverage: **88.06%**
- mean BASE: **-1.38981%**
- day-balanced BASE: **-1.57349%**
- positive rate: **23.73%**
- positive days: **0 / 5**

Cross-sectional leader audit:

- above-threshold clock buckets: **76**
- buckets with multiple candidates: **24**
- max simultaneous candidates: **4**
- attempts after leader filtering: **54**
- resolved: **46**
- mean BASE: **-1.24254%**
- day-balanced BASE: **-1.35891%**
- positive rate: **23.91%**
- positive days: **0 / 5**

The cross-sectional filter reduces some bad trades but does not reveal a profitable pocket.

## Interpretation

The phase features do not materially improve the ability to distinguish profitable from unprofitable executable states.

This suggests that the current problem may be upstream of the classifier:

1. the candidate/HOT population may contain too little post-cost opportunity;
2. the fixed 3-minute entry label may be a poor proxy for the path a skilled recurrent trader would exploit;
3. the available aggregated state variables may not contain enough information to predict which short-term path will follow.

Before building another model, quantify the opportunity ceiling.

## Next research boundary

Request218 should be an **oracle opportunity audit**, not another predictor.

On the same already-opened data:

- for each candidate episode, enumerate only executable watch states;
- compute the best realized BASE entry opportunity available inside the causal 1–5 minute watch window;
- separately compute gross and post-cost ceilings;
- measure how often an episode contains any post-cost positive state;
- measure the distribution of best achievable post-cost returns by day;
- compare exact-3m and event-time-3m ceilings;
- measure the gap between the best opportunity and the policies from Requests 208/215/217;
- do not use the oracle as a deployable feature or policy.

If the oracle ceiling is weak after costs, move upstream to candidate generation / opportunity horizon. If the oracle ceiling is strong, the bottleneck is representation/learnability and the next request should focus on richer raw sequence data rather than more feature algebra.
