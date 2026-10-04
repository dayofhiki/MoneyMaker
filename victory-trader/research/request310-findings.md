# R310 — positive short-clock increment, uncertainty is the remaining gate

Official run **37218791436**, artifact **11309014858**, source **7927cb2dfb90506d9a6b273c58bd0785c33408f0** (PR77). Same original 97 May5–8 episodes, 4,526 states, nine first-snapshot positives and R306 reachable BASE-net >=5% labels. Original 86 entries and all execution/stop/cost/cap rules remain frozen. No new market requests, June HOLD access, threshold tuning or trading model promotion.

The fixed primary clock-signal package improved all point-estimate ranking/probability endpoints versus background+coverage control, including same-day ranking, and improved two evaluable days with one tie. **Research gate remains FAILED: 7/8 checks passed, but the paired ticker-day within-day AUROC gain interval includes zero.** This is evidence supporting further research, not confirmed transport or executable profit.

## Fixed packages and first-snapshot results

All use unchanged R309 L2 logistic pipeline (C=.1, independent-episode weight sum, training-only weighted clipping/imputation/scaling, missing indicators). Train three May days, score the excluded fourth. First-snapshot training-prior Brier: **0.084378**. Weighted positive prevalence: **0.089182**.

| Arm | Fixed raw input package | Raw / processed dimensions | Pooled AUROC | Same-day AUROC | AP | Brier |
|---|---|---:|---:|---:|---:|---:|
| N | price, known cost, clock, previous-close return | 5 / 10 | 0.633328 | 0.659120 | 0.125424 | 0.080693 |
| Q control | N + print age and 10/30/120s activity coverage | 9 / 18 | 0.730140 | 0.739346 | 0.182935 | 0.078746 |
| M PRIMARY | Q + 19 fixed clock-signal features | 28 / 56 | **0.773723** | **0.800626** | **0.206803** | **0.077751** |
| W reference | exact full99 R309 pipeline replay | 99 / 198 | 0.734104 | 0.767318 | 0.175890 | 0.080544 |

The 19 added inputs are 10/30/120s endpoint return, signed efficiency, volume/transactions rate ratio, signed-close-volume proxy, previous-window high reclaim, plus 10-vs-30s acceleration. No feature was chosen from outcome associations or fitted coefficient rankings. Q already contains activity density and age. M-Q therefore estimates adding this whole directional/activity-rate package and its missing indicators, beyond those controls. N includes previous-close return, which is itself broad historical momentum; it is not a completely momentum-free baseline.

M-Q changes: pooled AUROC **+0.043583**, same-day AUROC **+0.061280**, AP **+0.023869**, Brier **-0.000996**. M's weighted predicted mean **0.089892** is close to observed **0.089182**. Average probability agreement does not establish individual calibration or a profitable threshold.

## Same-day endpoint and stability

Same-day AUROC includes ONLY positive-negative episode pairs within the same date, with the product of original equal-day/episode weights; ties count .5. This endpoint cannot improve solely by changing a held day's probability offset. Days without positives contribute no pair. Bootstrap preserves original row weights; repeated cluster draws are not new independent cases.

| Day | First positives / episodes | N AUROC | Q AUROC | M AUROC | W AUROC | M-Q |
|---|---:|---:|---:|---:|---:|---:|
| May5 | 0/21 | undefined | undefined | undefined | undefined | undefined |
| May6 | 5/29 | 0.616667 | 0.675000 | 0.783333 | 0.766667 | +0.108333 |
| May7 | 2/17 | 0.566667 | 0.800000 | 0.833333 | 0.800000 | +0.033333 |
| May8 | 2/30 | 0.910714 | 0.785714 | 0.785714 | 0.714286 | 0.000000 |

Within-day aggregate uses pair weights rather than simply averaging these three AUROCs. Point improvement on two days and a tie on one is encouraging, but each day's few positive episodes constrain certainty.

## Paired uncertainty, not another feature search

| First M-Q contrast | Ticker-day 95% interval | Day 95% interval |
|---|---|---|
| Same-day AUROC gain | **[-0.043838,0.195460]** | [0.000000,0.108333] |
| Pooled AUROC gain | [-0.021324,0.113249] | [0.029309,0.081847] |
| Brier difference | [-0.003168,0.001197] | [-0.003542,0.001590] |

There are only four day clusters and 97 ticker-day clusters with nine positives. Ranking-valid draws: 995/1,000 day resamples, 1,000/1,000 ticker-day resamples. Brier uses all draws. Day intervals do not override the preregistered ticker-day same-day gate. No removal of inconvenient cases or changed bootstrap/gate after fitting.

## What is learned about the bottleneck

R308 did not improve primary discrimination by emphasizing first states. R309's simpler pipeline improved pooled observability, with most gain from between-day comparisons. R310 now adds a positive same-day increment over price/cost/time/previous-close and coverage controls, so the observed benefit is not explained solely by day probability offsets or print coverage.

The control already gains same-day AUROC from N .659120 to Q .739346; some usable information concerns activity coverage. Adding clock signal gives a further gain to .800626. These are package comparisons within this regularized model, not causal contributions or individual indicator importance. Added dimensions also change the ridge geometry, and missingness indicators can carry coverage/history information. Signed-close-volume is not aggressor order flow. The conclusion is a promising signal package, not proof that each momentum feature is genuine or that raw short returns alone predict winners.

W is no longer the strongest first-snapshot point estimate, but it remains a replay reference, not a fallback selection rule. All fitted coefficients and preprocessing statistics are audited in JSON. No coefficients are turned into an entry gate or used to prune inputs after results.

## Later fixed observations

| Anchor after first saved decision | Episodes | Q pooled / same-day AUROC | M pooled / same-day AUROC |
|---|---:|---:|---:|
| +20s | 92 | 0.701885 / 0.767148 | 0.670407 / 0.740352 |
| +60s | 88 | 0.673963 / 0.671886 | 0.697424 / 0.685301 |
| +120s | 82 | 0.674064 / 0.664518 | 0.674007 / 0.662009 |

All-state pooled AUROC N/Q/M/W is .711469/.746069/.738084/.698112. The clock package is not a uniformly stronger model at every time. The first-state gain must not justify a fixed 20-second delay, forced hold or applying a first-state entry signal as an EXIT/HOLD controller. Timing interactions are unresolved and will not be tuned on these nine first winners.

## Next research boundary: freeze and broaden cases

The immediate bottleneck is uncertainty and independent early-case coverage, not a reason to hunt another C, seed, feature subset or confidence interval on May5–8. Freeze the Q/M input packages and exact pipeline before reading additional outcomes. A subsequent study should score separately defined May development candidates with a causal population/admission/snapshot schedule and all missing data reported, without fitting on their outcomes or searching for future winners. Verify upstream model fit dates and artifact provenance first; eliminate any future-label filtering of first observations or candidate coverage. Report within-day gains, positive counts, full population coverage and costs.

Existing request299-plan.md identifies May11–20 as possible additional development coverage and explicitly says those dates were previously explored elsewhere. They are NOT a pristine untouched holdout and must not be called independent confirmation of the overall research process. They can add episodes/days beyond these 97/4 when the fixed upstream/data contract is verified. Keep June HOLD and July–August final evaluation sealed. Do not silently bypass old gate-dependent promotion steps; additional coverage would be exploratory measurement, with no policy promotion. Missing more examples should be addressed with a preregistered causal population, not by admitting future winners.

The label still describes hindsight reachable net5 upside under the existing stop/cost schedule, not an executable selling policy. No model threshold or continuous trading policy was evaluated. Meaningful realized net profitability remains unestablished.

## Reproduction and verification

64 targeted tests and critical Ruff checks passed locally and official CI. New tests verify weighted same-day pair calculation, tie handling, invariance to date offsets, no same-day endpoint when all discrimination is between dates, original snapshot uniqueness and bootstrap duplicate handling, fixed causal packages, excluded held dates/held-label isolation, and replay mismatch/future-date rejection.

Official W replay error is below 1e-9; local replay is exactly zero. Official/local JSON is NOT byte/numerically exact: there are only tiny floating-point differences, with maximum scored probability difference **5.133e-12**, maximum report metric difference **1.653e-13**, and no nonnumeric/check/gate differences. Primary AUROC/AP/within-day rank results agree exactly; all numerical outputs are within 1e-9. Every fit converged (10–28 iterations), without adjustment. Persisted results/request310.json is the OFFICIAL JSON. Official artifact also preserves states, source SHA and logs.
