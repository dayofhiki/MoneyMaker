# R311 — momentum increment weakens in broader, sparse first-HOT observations

Official run **37223064559**, artifact **11311286614**, source **a6034adba43fbd02334834f68341a64951957ba6** ([PR79](https://github.com/dayofhiki/MoneyMaker/pull/79)). Frozen N/Q/M models fitted on the original 97 May5–8 episodes / 4,526 states; evaluation is an outcome-independent hash sample of 828 May11–20 full-HOT identities. No evaluation outcome fitting, threshold/entry tuning, June HOLD access or July–August access.

**Research gate FAILED: 8/10 checks pass.** M adds a positive same-day AUROC point increment over Q, but its paired interval still includes zero, and complete coverage is only 78.50%. Adding independent cases did not establish robust transport. M barely exceeds the simpler N same-day rank baseline. A post-result raw-data audit identifies an actionable measurement problem: nominal short-clock returns frequently bridge much older prices, while the existing current-print age feature is always zero at these event-triggered observations.

This is additional previously explored May development, not pristine validation. Training used a rich setup shortlist; evaluation used the broader upstream HOT population and a stricter first-observation contract. Those changes prevent attributing weaker performance solely to any one feature or disproving the original shortlist effect.

## Frozen first-observation results

N contains price, cost, HOT clock and previous-close return. Q adds print age and 10/30/120s activity fractions. M adds the same fixed 19 clock-signal inputs as R310. All use the unchanged L2 logistic pipeline, C=.1, seed20264605, training-only preprocessing and independent-episode weight sum97. Three fits converged in 12/13/17 iterations.

| Arm | Inputs | Pooled AUROC | Same-day AUROC | AP | Brier |
|---|---:|---:|---:|---:|---:|
| N background | 5 | **0.650922** | 0.638628 | 0.089947 | 0.056334 |
| Q coverage control | 9 | 0.627231 | 0.614336 | 0.081535 | 0.057182 |
| M clock signal | 28 | 0.642512 | **0.640588** | **0.099204** | **0.056287** |

650 complete cases contain **38 positive ticker-days across all eight dates**. Unweighted prevalence is 38/650=5.85%; original equal-day/episode weighted prevalence is **5.9431%**. The constant training-first prior Brier is **0.056784**. Mean weighted M probability is **7.2757%**, versus observed 5.9431%; average agreement alone cannot establish calibration.

M-Q: same-day AUROC **+0.026252**, pooled AUROC **+0.015281**, AP **+0.017669**, Brier **-0.000895**. M-N same-day gain is only **+0.001960**, pooled gain is **-0.008410**, and Brier difference is **-0.000047**. Thus the positive M-Q point difference partly reverses the deterioration from adding Q. It does not establish a large total improvement over price/cost/time/previous-close context. N already includes broad previous-close momentum and is not momentum-free.

| Paired contrast | Ticker-day 95% interval | Day 95% interval |
|---|---|---|
| M-Q same-day AUROC | **[-0.041224, 0.088153]** | [-0.068901, 0.107736] |
| M-Q pooled AUROC | [-0.048995, 0.070890] | [-0.039421, 0.069319] |
| M-Q Brier | [-0.002064, 0.000294] | [-0.001744, -0.000038] |
| M-N same-day AUROC | [-0.077136, 0.068691] | [-0.107455, 0.098365] |

All comparisons use 1,000 fixed paired draws; all rank draws are valid. Day Brier improvement does not override either failed preregistered check. Point estimates from R310 and R311 cannot be compared as paired effects: dates, candidate population, first-observation support and training/crossfit regime differ.

## Per-day stability and coverage

| Date | Sampled / complete | Positive cases | Q AUROC | M AUROC | M-Q |
|---|---:|---:|---:|---:|---:|
| May11 | 108 / 92 | 8 | 0.595238 | 0.650298 | +0.055060 |
| May12 | 87 / 65 | 4 | 0.516393 | 0.389344 | -0.127049 |
| May13 | 106 / 85 | 4 | 0.614198 | 0.617284 | +0.003086 |
| May14 | 112 / 87 | 3 | 0.396825 | 0.626984 | +0.230159 |
| May15 | 104 / 76 | 4 | 0.666667 | 0.770833 | +0.104167 |
| May18 | 102 / 82 | 4 | 0.538462 | 0.487179 | -0.051282 |
| May19 | 92 / 69 | 7 | 0.744240 | 0.771889 | +0.027650 |
| May20 | 117 / 94 | 4 | 0.700000 | 0.736111 | +0.036111 |

Six days improve and two deteriorate. Each date still has only 3–8 positive cases, with especially unstable May12/14 changes. No date or subgroup is removed, and no favorable day becomes a policy rule.

All **828 network requests succeeded without retries**; regular raw checkpoint has **859,466 seconds**. Available observations:679/828=82.00%; complete labels:650/828=78.50%. Coverage loss is observed sparse-print/fill support, not API acquisition failure.

| Missing condition, verified from raw bars | Cases |
|---|---:|
| First regular print starts at least five minutes after HOT | 143 |
| No regular print at/after HOT | 4 |
| HOT is exactly session close | 2 |
| Observation available, no entry print before cap | 4 |
| Entry exists, no terminal fill after submission | 25 |

The first three conditions constitute149 unavailable observations; the last two constitute29 incomplete labels. Incomplete cases have median HOT time358 minutes after open, compared with88 for complete cases. Unavailable cases have median135 minutes. Missingness is not plausibly neutral with respect to session time or activity. Scores/metrics remain conditional on complete observations; unknown cases are not relabeled as negatives. All828 identities and reasons remain in the official states artifact. Available unresolved cases are scored; unavailable cases have NaN model probabilities. No later complete/profitable observation replaces the first.

## Bottleneck: temporal meaning and observation support

The following are **post-result descriptive input audits**, not selected predictive subgroups or causal attribution. Raw training first-state features were independently replayed to establish that the historical raw reference matches the model inputs.

| Observation property | Original training first97 | New observed679 |
|---|---:|---:|
| Median HOT minutes after open | 16 | 95 |
| Median delay to completed first observation | 6s | 13s |
| Median observed seconds in last10/30/120s | 1 / 3 / 10 | 1 / 2 / 6 |
| Last10s contains at most one observed second | 55/97 (56.7%) | **505/679 (74.4%)** |
| Nominal10s return baseline median age | 18s | **32s** |
| Nominal10s return baseline older than20s | 43/97 | **423/677** available baselines |
| Nominal30s return baseline median age | 35s | 46s |
| Missing10s volume/transactions ratio or high-reclaim | 47.4% | **64.2%** |
| Missing30s volume/transactions ratio or high-reclaim | 11.3% | **38.4%** |
| Median next-entry-print delay, complete paths | 6s | **14s** |
| 90th percentile next-entry-print delay | 34.4s | **227.5s** |

`clock_features` uses the last completed close at/before the window start as the return baseline. Sparse trading means this can be much older than the nominal10/30/120s horizon. A nonmissing `momentum_return_10s` therefore does not imply a fresh ten-second price comparison. Neither Q's activity fraction nor its `momentum_gap_s` explicitly records the age of that baseline. Because the decision is triggered by a newly completed print, `momentum_gap_s` is **zero in every training and observed evaluation row**. It cannot reveal the preceding inter-print gap.

The signal package may consequently mix a price jump across a long silent interval with dense continuous acceleration. This is a concrete representation ambiguity verified from timestamps, not proof it caused the metric deterioration. Volume-rate/high-reclaim missingness also shifts sharply. Missing indicators partly expose availability, but they do not encode the actual elapsed time behind a finite return.

The all-state training distribution also differs from first observations. Its unweighted10s activity median is0.5 versus0.1 in new first observations. **After the actual equal-day/episode training weights, the median is0.2, not0.5** (30s0.1,120s0.075). Dense episodes must not be treated as thousands of independent cases, and raw row medians exaggerate the effective density difference. The diagnostic JSON preserves both weighted and unweighted descriptions. The broader candidate/time shift, sparse history and canonical-first support all remain relevant explanations to test.

This narrows the next intervention toward representing historical timestamp spacing and available evidence, rather than changing C, adding model capacity, or selecting winners. In a subsequent preregistered development study, explicitly encode prior-print/inter-print age, each return baseline's age and valid support, alongside direction/rate signals computed from completed observations. Compare unchanged clock inputs with a fixed availability-aware package on the **same causal population and first observations**, using day-excluded preprocessing/fitting. Retain every identity, disclose all missing paths, and keep any coverage decision separate from outcome-based filtering. Investigate matched population/training support before claiming broad generalization. No claim is made that feature engineering alone will restore performance.

These directions are not a new executed experiment, a tuned entry rule, a fixed waiting delay or authorization to open sealed evaluation. Freeze R311 first; register the next inputs/evaluation before fitting them. Event-triggered scores alone also leave continuous WAIT/ENTER/HOLD/EXIT behavior untested.

## Execution target still imposes a separate limit

82/650 complete cases have modeled same-price cost drag already crossing the -2.5% BASE stop. One of those82 nevertheless has reachable net5 because pending exits fill at a later print; this condition is not a universal impossibility filter. Of619 cases with complete whole-cap paths,43 have hindsight whole-horizon net5 but fail net5 under the absorbing stop. The whole-cap calculation is a diagnostic counterfactual, not an executable policy or a reason to relax costs/stops after viewing outcomes. Sparse next-print delays matter independently of direction prediction.

The label remains hindsight **reachable** net5 under fixed entry, stop, cost and cap rules. No sell policy, threshold, continuous controller or realized portfolio profit was evaluated. Research promotion remains false, with June HOLD and July–August sealed.

## Reproduction and integrity

Official CI: **72 targeted tests passed**, critical Ruff checks passed, and module execution succeeded. Persisted `results/request311.json` is byte-for-byte official output. `results/request311-diagnostics.json` is a separate post-result descriptive audit with SHA256s for the costly/raw/scored inputs. `analyze_request311.py` reproduces that audit without acquisition or evaluation-label fitting.

Audit checks: both source populations produce exactly the saved828 identities and context values; training/source context differences are all zero; raw bars are regular-session, unique and restricted to selected May ticker-days; all first observations and labels replay; all679 observed clock features replay; historical97 first clock features replay; unavailable probabilities remain missing. Frozen-model prediction replay maximum errors: N8.33e-17, Q/M1.11e-16. No entry endpoint, threshold, gate, date, hash sample, missing case or model package changed after outcomes.

To reproduce after downloading the declared official artifacts, run from `victory-trader` with the frozen research dependencies:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python research/analyze_request311.py \
  --official /absolute/path/request311 \
  --training /absolute/path/request310 \
  --training-raw /absolute/path/request306/request306-raw-seconds.parquet \
  --population /absolute/path/request233/request233-full-hot-states.parquet \
  --reference /absolute/path/request163/extended-history-economic-opportunity-v10.parquet \
  --output research/results/request311-diagnostics.json
```

Official [run/artifacts](https://github.com/dayofhiki/MoneyMaker/actions/runs/37223064559) preserve cohort, raw seconds, scored states, source SHA and execution log. R310/R233/R163 source run IDs are recorded in the plan and official JSON. The raw training reference is the original R306 artifact. The audit reads only allowlisted population input columns, and no June/July/August outcomes.
