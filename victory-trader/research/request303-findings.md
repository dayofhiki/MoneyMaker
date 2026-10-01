# R303 findings — weighted calibration helps exits, profitability still fails

Official run: https://github.com/dayofhiki/MoneyMaker/actions/runs/36835089902
Source: 2fe18cf6876d524d4ebb8ae3e6fcb5c4a4a53fa0. Artifact: 11148759001.
Execution succeeded; economic gate FAILED. All86 positions resolved, exact R302
replay and all saved predictions reproduced, weighted training residuals zero.
June HOLD remains sealed; no production promotion or subsequent dispatch.

## Economics

| Metric | A: R302 offset | B: weighted offset |
|---|---:|---:|
| Mean BASE net % | -1.449165 | -1.397021 |
| Median BASE net % | -1.203446 | -1.222123 |
| Positive entries | 6/86 | 8/86 |
| Positive days | 0/4 | 0/4 |
| Median hold seconds | 11 | 8 |
| First decision exits / eligible trajectories | 55/75 | 63/75 |
| Forced stops | 17 | 15 |
| CVaR5 % | -5.125685 | -5.125685 |
| Mean gross % | +0.093839 | +0.146481 |
| Mean cost drag pp | 1.543003 | 1.543501 |
| Mean excluding largest winner % | -1.523730 | -1.470973 |

B daily means May5–8: -2.092369,-1.012705,-1.848166,-0.981970%.
LIGHT -0.561225%, STRESS -3.633485%. Net +5/+10/+20 captures all zero.
Risk-compatible gross opportunities remain14/5/2; whole-path28/12/4. These
opportunities are retrospective diagnostics, not evidence of a learnable strategy.
48/86 exit fills wait more than3s; next-print references are not broker/quote fills.
Nine cost-only stop breaches remain included. Event means are not account returns.

B-A +0.052144pp; day bootstrap95% CI[+0.011838,+0.100820], ticker-day
CI[-0.008121,+0.120440]. All daily paired gains positive, but only four reused
days and ticker-day uncertainty crossing zero preclude a reliable improvement
claim. B-old +0.290973pp; ticker-day CI[-0.132197,+0.717924].

## Changes to trades

12 submissions earlier, none later; 11 realized returns change,7 better/4 worse.
Earlier submitted orders can still fill at the same next print, explaining the
12 versus11 count. Two stop exits become voluntary model exits. Examples:

| Day / ticker | A net % | B net % | Gain pp |
|---|---:|---:|---:|
| May6 SPCB | -0.542120 | +0.997756 | +1.539876 |
| May6 CXDO | -0.831200 | +0.479401 | +1.310601 |
| May5 POET | -2.560321 | -1.478962 | +1.081359 |
| May6 DGXX | +2.534564 | +1.740407 | -0.794158 |
| May7 STFS | -1.059189 | -1.829156 | -0.769967 |

The isolated shift improves the mean through changed exit timing; it does not
demonstrate stronger chart representation or learned sustained upside. Earlier
exits also sacrifice returns in some trades. Median and worst-tail metric do not
improve. Keep the correction as an explicit next-experiment baseline, not a
production promotion or silent rewrite of the historical shared fitter.

## Prediction versus deployed continuation

All-reachable weighted learned-reference rank +0.094381→+0.103243, but
self-policy rank remains negative: -0.073074→-0.038842. First-decision self rank
-0.078652→+0.045207 and policy-visited rank -0.012924→+0.060130. Positive changes
in those diagnostics are encouraging but weak and not independent validation.
Policy-visited sets differ (245 versus149 states), so their difference also reflects
changed visitation. Within-fold rank cannot change under a constant offset;
pooled rankings can change with differing fold offsets.

B reference/self label signs disagree24/75 first states,48/149 visited states,
4519/10619 reachable counterfactual states. Visited disagreement worsens from
34/245 while global disagreement shrinks; no claim of resolved consistency.
B predicted-HOLD visited group's weighted self advantage +0.357161pp versus
A -0.009930pp, with78 versus176 selected state rows. Selection and reused days
prevent interpreting this as a newly validated signal.

All75 eligible first decisions have causal chart return/range features available;
volume ratio58/75. Missing initial chart context is not the leading explanation.
66.19% of frozen pi0 targets are exactly zero;64.38% of reachable rows are aged
at least10min. The frozen teacher expires at entry+10min, while runtime learned
policy can continue to the common hot_t+60min/session cap. This intentional
contract, not leakage, is a substantial remaining learning/runtime discrepancy.

## Next experiment design priority (not executed)

R304 should isolate the frozen reference policy's expiry horizon. A: exact R303 B
replay. B PRIMARY: same fixed-reference 2% trailing teacher and mandatory stop,
but remove its separate entry+10min timer and use the SAME existing common cap
as runtime. This changes label construction only; refit the same model/features,
outer-day split, seeds, clipping, day/episode weights and corrected offset.
Do not impose minimum holds or force the runtime to wait60min: the next-observed
state sign decision remains voluntary, and stops/cap/pending exits are unchanged.

Rebuild teacher WAIT labels from next observed state forward. Mandatory stop is
absorbing and timer-stop ties retain risk priority. Cap submissions must execute
on first eligible regular print, without retrospective liquidation or deleting
gaps. Use May5–8 only, same86 entries and zero market acquisition. Assert pre-cap
labels no longer become zero solely because position age passed10min; zero
outcomes can still arise naturally. Test expiry boundaries, gap/cap/stop priority,
training-day exclusion and saved baseline reproduction before dispatch.

Report exact A reproduction, target support/zero fractions split before/after
10min, first/visited/self diagnostics, all-entry economics and paired intervals.
Keep the economic gates: positive BASE mean,>=3 positive days, positive mean
excluding largest trade, paired gains against A and old, resolution>=95%.
No held-day threshold selection, extra policy iteration or oracle-peak labels.
Aligning the reference cap still does not make that teacher optimal or identical
to the deployed policy, and cannot guarantee profits. If it fails, investigate
the fixed entries' feasible net upside versus exit-model learnability before
adding model complexity. June remains sealed until a complete policy is frozen.
