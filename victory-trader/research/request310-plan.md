# R310 — short-clock signal increment beyond background and coverage

Preregistered before fitting. Source official R309 run 37217714691: 97 original May5–8 episodes, 4,526 cash observations and unchanged R306 reachable BASE-net >=5% labels. Same 86 frozen entries. June HOLD sealed; zero acquisition. No entry, exit, stop, cost, threshold or policy change.

R309's linear control W improved pooled first AUROC .597->.734, but 91.46% of the gain was in between-day pair comparisons. Within-day pair AUROC improved .718->.767. Price/cost/time and momentum were mixed. R310 tests a fixed added feature package, with within-day first-snapshot ranking a primary endpoint. Reusing these days and choosing this research direction after R309 makes the experiment exploratory even if its gate passes.

## Fixed arms, same training and pipeline

- N, background5: log_current_price, minutes_since_open, minutes_to_close, return_from_previous_close_pct, known_cost_drag_pct. Previous-close return is broad past momentum/context; this is not a claim of a wholly momentum-free control.
- Q, background+coverage9: N plus momentum_gap_s and momentum_activity_fraction_{10,30,120}s. Q accounts for observed print age/density so that signal increments are not attributed solely to coverage or missing-history information.
- M PRIMARY, background+coverage+clock signal28: Q plus all remaining R306 CLOCK_FEATURES, deduplicated in fixed order. The 19 additional features are return, signed efficiency, volume/transactions rate ratio, signed-close-volume proxy and prior-high reclaim for 10/30/120s, plus 10-vs-30s acceleration. They are one preregistered package; no outcome-based feature/scale selection.
- W, full99 exact R309 replay: reference/mechanism control, not an alternative primary arm to deploy.

Each logistic model calls the UNCHANGED R309 fit_linear and predict_linear: training-only weighted 1%/99% clipping, weighted median imputation, weighted scaling and missing indicators; C=.1, L2, solver lbfgs, max_iter2000, tol1e-8, weight sum=independent training episode count. No class balancing, interactions, calibration, sign constraints or inversion. Same all-state day/episode weights; no first-only or time reweighting. Fit three days and score the excluded day, seeds20264100+fold*100+5. W replay error >1e-9 aborts. Convergence failure aborts without retuning.

Input allowlist must exactly match R309's original99 and R306 causal baseline+clock. Never include saved scores, target/label/execution/future fields, ticker/day identity or own-day labels in X. Completed second bars only, inherited with their original missing data. Coverage features and missing indicators cannot be equated to actual aggressor flow.

## Evaluation and uncertainty

At first original saved cash observation (97 independent episodes,9 positives), primary within-day AUROC compares only positive-negative episode pairs from the SAME day. Pair weight is the product of original equal-day/episode evaluation weights; ties count .5. Aggregate all eligible within-day pairs; a day without positives contributes no pair. Thus this endpoint cannot improve solely through held-day probability offsets. Also report pooled AUROC/AP/Brier and first-snapshot training-only prior Brier, per-day first metrics and positive counts. Secondary fixed snapshots +20/+60/+120 seconds retain original gaps/availability; no optimal delay or forced hold. All-state metrics are descriptive without a within-day episode endpoint.

Paired bootstrap: 1,000 common draws, day and ticker-day clusters, seeds20264500+cluster_type. Preserve original evaluation row weights within each resampled cluster; repeated rows in bootstrap are draws, not new independent examples. Report M-Q primary, Q-N coverage, M-N total and W-Q full-package contrasts for pooled AUROC, within-day AUROC and Brier. Single-class/no-within-pair draws are omitted only for the relevant ranking endpoint and counted. No gate or arm change after results.

Research gate requires exact W replay; M first within-day AUROC, pooled AUROC and AP better than Q; M Brier below Q and the first-training-snapshot prior; M per-day first AUROC improves by >1e-9 in at least two evaluable days; positive lower bound for paired ticker-day M-Q within-day AUROC gain. Gate passage is development-only, not promotion eligible. W/Q/N are controls, not fallback selections if M fails.

Four reused days, nine first positives, correlated state histories, upstream OOF-conditioned shortlist not nested, overlapping feature/missing indicators and ridge geometry constrain interpretation. Increment estimates concern adding this feature PACKAGE to this fixed pipeline, not causal feature attribution, unique contribution of each indicator or executable profitability. If full99 retains a gain while clock28 does not, separately motivated raw-path/context representation and independent candidate coverage must be considered; do not chase a feature subset on these nine winners or open June.
