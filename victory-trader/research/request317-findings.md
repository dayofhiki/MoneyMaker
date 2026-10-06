# R317: continuous states partly repair first-only training, without a momentum increment

R317 follows the main R315 training-support question. R316/316B remain separate
draft event studies. The local plan commit f62ce70 preceded state construction,
new labels and fits; remote preregistration fb334bfc68eb712c173a27c2c010147247cb69be
precedes official code790b831683ce00dc4d3a4f54dab4db8eead10feb.
No market requests, evaluation-label updates, policy, threshold/window search,
June HOLD or July-August access. All May results remain reused development.

## Support and fixed contracts

The original407 hash-selected May5-8 broad HOT episodes yield347 observable
episodes and9,325 completed clocks within the inherited300s window. Sixty
episodes have no eligible print and remain in the ledger. Of those clocks,
8,792 have complete next-open/BASE-cost/-2.5%-stop labels from336 episodes;
533 states stay censored. The clock manifest is written before labeling.

| Training arm | Complete states | Independent episodes | Ever-positive episodes |
|---|---:|---:|---:|
| B broad first-only M28 |334|334|35|
| C broad continuous M28 |8,792|336|62|
| Q controls9 on identical C rows |8,792|336|62|
| H continuous M28, matched B episodes |8,733|334|62|

Only two episodes gain a complete label after a censored first label. H and B
have exactly the same334 episode identities; their first features/clocks/labels
replay. Thus H-B tests observation schedule with episode membership held fixed.
C and Q share exactly the same rows and independent weights. Equal-day and
equal-episode weighting has effective sum336 for C/Q and334 for H/B. The9,325
clocks are correlated observations, not9,325 independent trading opportunities.

All states are alternative cash decisions; labels do not simulate a position
being stopped and reentered. Each state uses only completed aggregates, while
HOT background context stays frozen at original HOT semantics. Learning later
training phases is tested at the original FIRST evaluation clock; R317 does
not test a later-entry or continuous executed trading policy.

## Same650 complete first-clock evaluation cases

All828 original evaluation identities,679 available scores and650 complete
labels are retained. R and B probabilities replay their original saved scores
with zero local error. The previous Q refit in R315 is explicitly distinct from
R311 Q; the integrity guard permits only that prior score-column replacement
and still verifies all other original inputs, labels and frozen scores.

| Fixed model | Same-day AUROC | Pooled AUROC | AP | Brier | Own-prior Brier |
|---|---:|---:|---:|---:|---:|
| R original selected repeated-state M |.640588|.642512|.099204|.056287|.057282|
| B broad first-only M |.569407|.566615|.087382|.062642|.057957|
| C broad continuous M |.609553|.603009|.085894|.058848|.056820|
| Q same continuous support, controls only |.639917|.630069|.142359|.056895|.056820|
| H continuous M, same334 B episodes |.610951|.603179|.085840|.058843|.056848|

C-B same-day improvement+.040146 has ticker-day95% CI[-.010044,+.087432]
and day CI[-.032787,+.082528]. H-B+.041544 has ticker-day CI
[-.008294,+.088909] and day CI[-.032211,+.083026]. Both cross zero: the
point recovery is not a robust rank improvement. C beats B on six/eight dates,
but AP is slightly lower. Probability error does improve: C-B Brier difference
-.003793 has ticker-day CI[-.005804,-.001692] and day CI
[-.004692,-.002840]; matched H-B is nearly identical.

C-Q same-day difference-.030363 has ticker-day CI[-.075518,+.014344]
and day CI[-.074933,+.014079]. Rank harm alone is inconclusive, but AP harm
is supported: difference-.056465 with ticker-day CI[-.121165,-.009551]
and day CI[-.122506,-.010464]. Brier is worse by+.001953 with ticker-day
CI[+.000636,+.003450] and day CI[+.000328,+.003404]. Thus directional
inputs do not establish a benefit even on restored continuous training support.

C remains below R in point same-day AUROC and AP, with rank intervals crossing
zero. C also has worse Brier than its own prior. Q does not beat its own prior
Brier either. No model is selected as a fallback. All1,000 rank draws are valid
under both preregistered cluster schemes; they condition on these fixed fits
and do not quantify fitting uncertainty or independent generalization.

Only3/10 checks pass: positive episode support, positive-date support, and
majority-date improvement over B. Coverage remains650/828=78.50%, below90%.
No criterion is relaxed, and no missing outcome becomes a negative example.

## Chronological training diagnostic

Earlier-day continuous C fits1,465/5,655/7,274 states from80/175/259 episodes
for May6/7/8 respectively. Each transform and fitted label excludes the held
and later days. On the255 complete held first states, same-day AUROC is
B .740351, C .789351, Q .800464, H .790645. C Brier .087331 versus
B .087413 and Q .088293. These are different dates/support/fits from the650
evaluation, not an identified estimate of temporal deterioration. No tuning
or model choice is made from these diagnostic results.

## Decision and next question

Restoring continuous observations adds positive phases and improves probability
error relative to first-only broad training. It is not sufficient to recover the
original model or establish directional increment. The near-identical C and H
results make the two newly eligible episodes an unlikely explanation for this
point recovery. This narrows the failure of the current fixed representation
and fit; it does not refute momentum trading in general.

The next separately registered question should be whether direction/activity
signals need context interactions, rather than another observation-count/window
expansion. A fixed limited-capacity interaction/nonlinear M versus SAME-row Q
comparison, with equally expressive Q and linear C comparators, can distinguish
directional information from a general capacity gain. Any feature/architecture
choices must be frozen before new evaluation, with chronological training checks,
unchanged costs/stops/missingness and the sealed periods still closed. Interaction
failure is a hypothesis, not a cause established by R317. No policy, specialist,
router or deployment follows this development result.

## Execution and verification

Official cached replay: [run37472421963](https://github.com/dayofhiki/MoneyMaker/actions/runs/37472421963)
**succeeded**, source790b831683ce00dc4d3a4f54dab4db8eead10feb,
artifact11417253645. ZIP SHA256 verifies:
`a3dd142e7e3f7e072827da984345ee17df316306cbf4f4cfbe73b672d1096bfb`.
All3,574 JSON numeric fields reproduce within1e-9; actual maximum error
2.09e-14. JSON bytes differ and are not claimed identical. All9,325 clocks,
407 ledger rows,9,325 training states,828 evaluation states and305 chronological
states match identities, clocks, label values, missingness and probability pair
ranks; maximum numerical error across outputs is2.31e-14. Label/clock manifests
and original input hashes verify. request317-reproduction.json records the audit;
request317.json is the exact official JSON, with local output separately marked.

GitHub full code CI: **1,041 tests pass**, both test and flatfiles jobs succeed.
Official cached execution and final local targeted validation: **54 tests pass**,
with critical Ruff. Local pre-correction full suite also passed1,040 tests.
The correction changes only a guard
that previously compared two legitimately different saved Q fits; no feature,
label, model parameter, observation clock, sample or result gate changes.
The audit script compares JSON numbers within1e-9, all flags/identities/missingness
and complete probability pair ranks without fits. The one-shot branch workflow
is removed after success so documentation updates do not launch experiments.
research-request.json and main remain unchanged; R317 is published as a draft.
