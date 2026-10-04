# R309 — fixed compact momentum representation and model capacity

Preregistered before fitting. Reuse official R306 run 37211254516 only: original 97 May5–8 cash episodes, 4,526 states, same reachable BASE-net >=5% target and causal source features. June HOLD stays sealed; zero new market requests. Keep the original 86 entries and execution/cost/absorbing stop/cap policy frozen. This is observability research, not a strategy backtest.

R308 did not support a substantial first-snapshot weighting intervention as sufficient. R309 separates feature compression from model architecture with a 2x2 comparison using identical all-state day/episode weights; one additional first-only arm diagnoses correlated-history transport. No arm is selected after results.

## Fixed causal input package

Ten existing, outcome-independently specified features describe the current 30-second state plus immediate acceleration and feasibility:

1. momentum_return_30s — direction/size of movement
2. momentum_efficiency_30s — signed movement efficiency
3. momentum_volume_rate_ratio_30s — volume activity versus preceding window
4. momentum_transactions_rate_ratio_30s — print activity versus preceding window
5. momentum_signed_volume_proxy_30s — sign-of-close-change proxy, not trade aggressor flow
6. momentum_reclaim_prior_high_30s — distance to preceding-window high
7. momentum_activity_fraction_30s — completed-second activity coverage
8. momentum_gap_s — age of last completed print
9. known_cost_drag_pct — known roundtrip drag at current price
10. momentum_acceleration_10_30 — immediate change in movement rate

All are previously saved R306 features computed from COMPLETED seconds at the decision. No new future feature, outcome-based univariate selection, saved learned score, ticker/day identifier or execution label enters X. No feature interactions, sign restrictions or score inversion.

## Fixed arms

| Arm | Representation | Model | Training rows |
|---|---|---|---|
| U | original 99 inputs | original R306 tree | all |
| V | fixed 10 inputs | same tree | all |
| W | original 99 inputs | regularized logistic | all |
| X PRIMARY | fixed 10 inputs | same logistic | all |
| F secondary | fixed 10 inputs | same logistic | first saved row per episode |

U must replay saved R306 C to <=1e-9 or abort. U/V use unchanged R306 fitter, tree parameters and seeds. Each model trains on other three days and scores the held-out fourth. First-only F provides 67–80 rows, with just 4 or 7 or 9 first positives, audited per fold. No additional samples are created.

Logistic preprocessing is fitted on its TRAINING rows only, without labels: per-feature weighted 1%/99% clipping and weighted median missing-value imputation, followed by weighted mean/standard-deviation scaling; append one binary missing indicator per feature. All-training-missing dimensions use 0 with scale1 and are audited. Held-out values cannot alter any transform. Both raw and processed dimensions are reported. Comparisons U vs W / V vs X estimate the whole linear model+preprocessing package, not architecture in isolation; X vs W estimates the predeclared feature package within that pipeline. U/V preserve native tree NaN handling.

LogisticRegression solver=lbfgs, C=.1, l1_ratio=0 (L2), fit_intercept=True, max_iter=2000, tol=1e-8, no class balancing, no calibration. Day/episode weights are rescaled to sum to the number of independent training episodes, so thousands of correlated rows do not weaken regularization versus F. C is fixed before fitting; there is no CV/search. Seed remains 20264100+outer_fold*100+5. Non-convergence aborts, rather than tuning solver/regularization after results. The legacy tree weights stay normalized to mean1 for exact replay.

## Fixed endpoints and gate

Primary first saved decision: day/episode weighted AUROC, AP, Brier and first-training-snapshot prior Brier. Secondary original anchors +20/+60/+120 seconds and all-state metrics; no prescribed waiting or minimum holding. Report per-day first AUROC and paired cluster bootstrap AUROC/Brier differences versus U, using 1,000 draws, day and ticker-day clusters, seeds20264400+cluster_type. Single-class AUROC draws are omitted and counted. Report X-U, X-V and X-W paired comparisons; the mechanism controls do not become replacement primary arms if X fails.

Research gate: exact U replay; X first AUROC/AP better than U, X first Brier better than U and the first-snapshot training prior; first X AUROC improves by >1e-9 on at least two evaluable days; positive lower bound for ticker-day X-U AUROC interval. Also report whether X beats V and W, without treating that as a causal feature-importance claim. F is secondary and not eligible for post-result promotion. Even a pass remains development-only: four reused days, nine first winners and upstream OOF-conditioned shortlist not nested. No post-result feature, C, preprocessing, arm, seed, label, threshold, calibration or execution tuning.

If this package fails, do not infer that all momentum is absent or scan for a lucky architecture on these nine winners. Use this result to preregister additional independent May candidate coverage with a causal selection population, or a separately motivated signal measurement intervention. Do not open June or acquire cases chosen by future gains.
