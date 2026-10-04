# R308 — weighting is not sufficient to recover initial momentum

Status: official experiment complete. R308 run **37213959708**, artifact **11307871895**, source **df1cea940ada51a92a39dcf1d533c27b83a8a711** (PR73). Source R306 run 37211254516. Same 97 May5–8 episodes, 4,526 cash observations, 99 causal inputs and reachable BASE-net >=5% labels. First-snapshot support is 97 episodes with 9 positives (day counts 0/21, 5/29, 2/17, 2/30). No June HOLD, market requests, policy changes or threshold selection.

## Primary first-decision results

Equal day/episode evaluation weights. AUROC/AP higher is better; Brier lower is better.

| Fixed arm | AUROC | AP | Brier | Mean predicted probability |
|---|---:|---:|---:|---:|
| U: exact R306 C replay | 0.597127 | 0.127366 | 0.103026 | 0.047866 |
| T: clock-phase balance | 0.653820 | 0.133175 | 0.106272 | 0.047578 |
| E: primary first-snapshot mixture | 0.579829 | 0.134432 | 0.092786 | 0.032772 |

First-snapshot training-prior Brier is **0.084378**, better than all three models. Weighted observed prevalence is **0.089182**. E reduces some large errors, but further lowers mean predictions despite average underprediction. Improved Brier alone therefore does not show that the model learned early momentum. Research gate **FAILED**: E loses AUROC, does not beat the first training prior, improves only one evaluable day, and has no positive paired-AUROC interval.

## The intervention was substantial

Rows per episode: median 16, mean 46.66, range 2–279. Across the full development population, U gives first observations 11.08% of learning mass and >=120s observations 39.62%. T changes these to 15.10% and 22.25%; E changes them to 57.55% and 11.13%. These are descriptive full-population exposures, not a fit using all four days. Each actual outer training fold is separately audited in JSON. Equal mass across AVAILABLE phases need not give exactly 25% population mass to each phase when episodes lack phases.

Increasing first-observation mass more than fivefold did not improve the primary AUROC. This weakens the specific hypothesis that later-observation weighting alone is the sufficient bottleneck. It does not disprove all possible weighting choices, prove an absence of early signal, or authorize another fraction search on these results.

## Cross-day stability and uncertainty

| Held-out day | U AUROC | T AUROC | E AUROC |
|---|---:|---:|---:|
| May 5 | undefined (no positives) | undefined | undefined |
| May 6 | 0.658333 | 0.791667 | 0.683333 |
| May 7 | 0.866667 | 0.800000 | 0.800000 |
| May 8 | 0.607143 | 0.464286 | 0.607143 |

May8 E and U are equal to numerical precision. The T aggregate gain is not stable within days and cannot be selected as a replacement after the primary E failure. Paired ticker-day 95% AUROC gain intervals: T-U **[-0.022600, 0.167054]**, E-U **[-0.122886, 0.083025]**. Brier differences: T-U **[-0.002217, 0.009050]**, E-U **[-0.023828, 0.001735]**. Day-bootstrap intervals are separately retained; four reused days cannot establish robust transport.

The pre-existing descriptive >=0.5 probability bin still contains no first-snapshot positives: U 4/0, T 4/0, E 2/0 (episodes/positives). The nine positives' median scores are U 0.019514, T 0.012713, E 0.003112. These are diagnostics only, not a chosen trading threshold or an inverse-score rule.

## Secondary fixed clocks

| Snapshot | U AUROC | T AUROC | E AUROC | Available episodes |
|---|---:|---:|---:|---:|
| +20s | 0.650516 | 0.507063 | 0.592291 | 92 |
| +60s | 0.586023 | 0.575390 | 0.587169 | 88 |
| +120s | 0.578327 | 0.491648 | 0.526287 | 82 |

Clock anchors start at the first saved cash decision and use the first original observation at/after the anchor. Gaps and missing snapshots are preserved. These are not optimal waiting times. All-state AUROC U/T/E is 0.638654/0.620219/0.636462; the 4,526 observations are correlated and do not add independent winners.

## Refined bottleneck and next research boundary

R306 exposed an objective/feature mismatch, R307 showed that probability calibration cannot repair unstable ranking, and R308 shows that a substantial first-state weight intervention is insufficient. The remaining unresolved question is whether a constrained representation can transfer across independent early episodes, or whether the current 9 positive examples do not cover the situations needed. The present experiment cannot separate those causes; 99-dimensional nonlinear trees also retain row-count histogram and minimum-leaf constraints despite weighting.

Do not force holding, relax the absorbing BASE-net -2.5% stop, modify execution/cost assumptions, select T after E fails, or retune on June. The next controlled comparison should fix a small causal pressure/efficiency representation and a strongly regularized classifier before fitting; use first-snapshot evaluation and expose independent positive counts. Any subsequent data expansion must preregister time coverage and a causal candidate population, not search for future winners. Keep the original 86 entries and 97 episode reference as frozen benchmarks. This is no evidence of an executable profitable policy.

## Verification

53 targeted tests and critical Ruff checks passed. Tests check day/episode/phase mass, invariance of phase mass to denser late sampling, label-independent weights, missing phases, duplicate decision rejection, held-day label isolation, future-day rejection and replay mismatch. U exactly reproduced every saved R306 C score (maximum absolute difference 0 in all folds). No historical fitter or earlier experiment was altered. See request308-plan.md and results/request308.json for the fixed protocol and full audit.

Official JSON exactly equals local JSON. All 4,526 predictions for U, T and E agree with absolute error 0. Official module and CI tests succeeded; the research gate remains FAILED. The full result is persisted in results/request308.json; the official artifact also contains the scored states and source/log provenance.
