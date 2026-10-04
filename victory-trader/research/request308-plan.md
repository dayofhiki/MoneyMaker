# R308 — initial momentum training-weight bottleneck

Preregistered before fitting. Reuse the full official R306 artifact (run 37211254516): 97 original May5–8 cash candidate episodes, 4,526 observations, unchanged reachable BASE-net >=5% label and 99 causal features. June HOLD stays sealed. No market requests, new candidates, policy or execution changes. The 86 frozen reference entries stay unchanged; this is an observability experiment, not a trading backtest.

R307 improved Brier but destroyed discrimination; two training-only calibration slopes were zero. It did not justify probability inversion. R308 tests whether equal episode mass spread over many later observations suppresses first-decision learning. It does not create new independent examples or establish that weighting is the cause.

## Fixed arms

- U: reproduce R306 C exactly using its uniform within-episode state weights.
- T: equal mass per AVAILABLE clock phase [0,20), [20,60), [60,120), [120,infinity) seconds from the first saved decision. Split each phase's mass uniformly over its states; each episode and each day retain equal mass. Missing phases receive no synthetic rows.
- E, PRIMARY: fixed mixture of 0.5 point mass on the first saved observation plus 0.5 T distribution. Thus the first row gets slightly more than half an episode's mass; the exact share is audited. Fractions and boundaries will not be selected after results.

All arms use all original rows, R306 features/labels, the same HistGradientBoostingClassifier parameters (learning_rate=.04, max_iter=220, max_leaf_nodes=15, min_samples_leaf=75, L2=3, early_stopping=False), and the exact R306 seed 20264100+outer_fold*100+5. Train on three May days, score the excluded fourth. Replay mismatch >1e-9 aborts. The historical R306 fitter is unchanged. Weight selection depends only on episode IDs and decision clocks, never future labels, prices or scores. Tree histogram bins and minimum leaf sizes still count correlated rows; this is a weight intervention only.

## Fixed assessment

Primary endpoint: original first saved decision, reachable BASE-net >=5%, equal day/episode evaluation weights. Compare weighted AUROC, AP and Brier. A separate baseline uses the FIRST-snapshot label prevalence of the three training days; all-state priors are also reported. Secondary fixed snapshots are +20/+60/+120 seconds, selecting the first available original row at/after the anchor; these are not forced holding or prescribed entry delays. All-state scores are descriptive.

Paired day and ticker-day cluster bootstrap: 1,000 fixed-seed draws, seed 20264300+cluster_kind. Report E/T minus U AUROC and Brier CIs. Single-class draws are omitted only for AUROC and counted. Four reused days make day intervals fragile.

Research gate requires exact U replay, E better than U on first AUROC/AP/Brier, E Brier better than the first-snapshot training prior, E first AUROC improving at least two of three evaluable days, and positive lower bound for ticker-day paired E-U AUROC CI. Even a pass is development-only and not promotion eligible. T is a mechanism control, not an alternative to select if E fails. No feature, target, seed, phase, fraction, threshold, tree, calibration or policy changes after results.

Interpret results with nine first-snapshot winners, 97 episodes and four reused days, upstream OOF-conditioned candidate selection that is not nested, costs/absorbing stop limitations, and 1-second aggregated data rather than NBBO/broker fill truth. Failure will narrow the next step toward independent early candidate coverage and representation, without opening June or choosing future winners.
