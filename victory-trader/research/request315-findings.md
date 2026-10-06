# R315: broader first-state training does not repair momentum transport

Official Python3.11 [run37450538746](https://github.com/dayofhiki/MoneyMaker/actions/runs/37450538746),
artifact11405473272, source `c271fbab9c3567c470fc852ef80c39fe36116713`
([PR86](https://github.com/dayofhiki/MoneyMaker/pull/86)) succeeded. Local plan
`037142b` preceded new labels, fits and acquisition. Original costs/stop,
M28/Q9 pipeline and original828 evaluation identities remain fixed. Official
102 targeted tests and full PR86 CI1,029 tests pass. No sealed dates or policy.

## What was actually changed

Two independent official sources agree on all1,602 May5–8 HOT identities and
allowed HOT context. The fixed R311 hash samples407. Union with original97
selected identities is478, overlap26, persisted before acquisition. Reuse97
original raw histories;381 declared training-only requests,0 retries/failures,
580,944 regular rows. No evaluation acquisition. Save all407 broad identities,
including60 without first observation and13 unresolved entry/terminal labels;
334 complete first states enter B/Q fits,35 net5-positive episodes, weighted
prior .104796. Selected97 have97 complete first states,9 positives, prior .089182.

**All97 canonical selected clocks equal the original first saved clocks**, with
unchanged first labels and zero positive transitions. Thus suspected first-clock
misalignment is not present on this support. S versus R changes the training
state schedule (97 first states versus4,526 repeated states), not those first
clocks/labels. Repeated-state training includes30 episodes positive somewhere
among its states, versus9 at first observation; no claim that these are the
same target distribution. B versus S changes population and sample size at the
same first-observation contract. All comparisons remain development interventions,
not identified causal effects.

## Same650 final-label evaluation

| Fixed model | Same-day AUROC | Pooled AUROC | AP | Brier | Own training-prior Brier |
|---|---:|---:|---:|---:|---:|
| R original M,4,526 states /97 episodes |.640588|.642512|.099204|.056287|.057282|
| S selected97 canonical first states, M28 |.605157|.598968|.078381|.057621|.056784|
| B broad334 canonical first states, M28 |.569407|.566615|.087382|.062642|.057957|
| Q same broad334 first states, controls9 |.627493|.616882|.123406|.059588|.057957|

B−S same-day difference−.035750, ticker-day95% CI[−.134945,+.071552];
B−R−.071181, CI[−.190634,+.035818]. Those rank contrasts do not establish
improvement or resolved harm. Brier B−S+.005021, CI[+.000286,+.009238], and
B−R+.006355, CI[+.001524,+.010951], indicate worse fixed-score probability
error on this cohort.

**Momentum increment B−Q is adverse:** same-day−.058086, ticker-day CI
[−.109802,−.008667], day CI[−.119977,−.002090]. AP difference−.036025,
ticker-day CI[−.079105,−.000105]; Brier+.003054,
CI[+.000926,+.005461]. Adding the19 directional inputs to the same9 controls
does not help this frozen broad first-state fit. This is evidence against this
particular representation/fit on reused May, not against momentum as a market
phenomenon. Q improves B on six/eight date AUCs; B beats S on four/eight,
not a majority. Q itself has worse Brier than its prior and R, so this is not
a deployable Q promotion or a selected fallback.

Only2/9 fixed checks pass. Evaluation coverage650/828=78.50% and training broad
coverage334/407=82.06% remain below90%; no missing label is set negative and no
criterion is relaxed. All679 observed evaluation cases receive scores, regardless
of labels; all828 identities remain in the scored checkpoint.

## Chronological training diagnostic and interpretation

May6 fits79 May5 labeled rows; May7 fits174 May5–6 rows; May8 fits258 May5–7.
Each transform/model excludes current and later days; May5 is unscored. Of305
held identities,266 observed and255 labeled,27 positives. B same-day AUROC
.740351, Q .813054; Brier .087413 versus Q .084976 and prior .095323.
These refer to different dates/cases/fits from the650 evaluation and are not
paired temporal degradation estimates or independent confirmation. Stronger
within-training point results do not license claiming later May transport.

R315 rules out broader canonical first-state training as a sufficient repair.
It also shows that the original97 first clocks were already canonical. The
remaining distinction between first-only and repeated causal states is material;
discarding later training states loses observation phases. The known original
count is30 ever-positive episodes versus9 first-positive episodes.
Do not infer that increasing state count alone will help or manufacture independent
examples by counting seconds as episodes.

Next separately registered research should use the newly checkpointed broad
raw histories to compare first-only versus all completed causal observation
states under a fixed early window, keeping day/episode weights and the original
evaluation clocks/labels. This restores continuous-observation learning support
and tests the directional increment on exactly matched rows, rather than finding
a profitable delay or weakening stops/costs. Chronological checks remain required;
no new window, threshold or model is selected from these outcomes. June HOLD
and July–August stay sealed.

## Offline reproduction

All2,084 JSON numeric fields agree exactly, as do all nonnumeric fields and gate flags. Official and offline JSON bytes are identical. All828 scored evaluation rows,478 training states,478 membership rows and580,944 raw-second rows have zero numeric error, identical missingness and probability ranks. Original acquisition checkpoint hashes verify. Exact official JSON and the separately marked identical offline JSON are retained with the reproduction audit. The offline replay makes0 market requests.

An optional boolean checkpoint field stores
in-memory NaN as Arrow null; the strict original guard rejected this representation
change. The replay-only fix normalizes missing representations without changing
missing positions or observed values; a parquet roundtrip regression also checks
that changed values/missingness still fail. Fits, features, labels, cohort and
evaluation rules are untouched. Post-fix local103 targeted tests pass. No extra
market acquisition or duplicate official experiment is needed for this replay.
