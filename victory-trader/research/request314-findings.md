# R314: gross opportunity decomposition does not improve net ranking

Official Python3.11 run [37448072456](https://github.com/dayofhiki/MoneyMaker/actions/runs/37448072456), artifact11403568453, source `f3b07c5a7c9f06a531dac3934e6c8a3f82225fcb` ([PR84](https://github.com/dayofhiki/MoneyMaker/pull/84)) succeeded. Plan committed locally at
`e77e8d1` before code at `1037d69` and before new stage labels/fits/evaluation.
Fixed M28, original costs/absorbing stop and pending fills, no search, sealed
dates or market requests. Critical Ruff and93 official targeted tests pass; full PR84 CI has1,020 passing tests.

## Opportunities lost between gross rise and stopped net

The 828 original May identities remain intact: 679 observations, 650 complete
final labels, 619 also complete whole-horizon labels. The 31 whole-censored but
final-complete identities stay in the primary 650-case evaluation. All 38 final
net5 positives happen to have complete whole labels; this does not resolve the
31 unknown whole paths. No missing label becomes a negative.

| Common619 opportunities at original first observation | Positive cases |
|---|---:|
| Whole gross>=5 from last observed close (counterfactual) |131|
| Whole gross>=5 from actual next-open entry |124|
| Whole BASE net>=5 from actual entry, no absorbing stop |81|
| Original stopped BASE net>=5 |38|

Observed-close versus actual-entry repricing loses14 and gains7 opportunities;
it is not a monotone funnel. Costs remove43 of the124 actual gross opportunities;
the unchanged absorbing stop removes another43 of the81 whole-net opportunities.
These are hindsight opportunity ceilings, not realized returns or causal proof
that eliminating costs/stops is a useful executable intervention. Whole-cap
diagnostics are conditioned on complete future labels. No fill at observed
close or earlier price is fabricated.

All4,526 original97-episode training states have complete whole labels, so D
and original R receive exactly the same support and predictions. Two seeds
have no effect for this deterministic solver. Conditional H retains original
common training row weights; no episode/day renormalization. Its prior times
gross prior exactly equals the direct final-net prior. Inference uses G×H on
every available observation, never an evaluation gross-label gate.

| Frozen model on original650 final labels | Same-day AUROC | Pooled AUROC | AP | Brier |
|---|---:|---:|---:|---:|
| R original M / D common-support direct |.640588|.642512|.099204|.056287|
| P gross probability × conditional net conversion |.633855|.639482|.098404|.056628|

Constant direct training-prior Brier .057282. P beats this constant baseline
but is worse than D on all listed point metrics. P−D same-day AUROC -.006733,
ticker-day95% CI[-.034054,.019323]; Brier difference+.000342,
CI[-.000350,.001079]. Neither improvement nor a statistically resolved harm
is established. Only2/8 gate checks pass; development and promotion-ineligible.
Inherited complete coverage650/828=78.50% still fails90%, with no relaxation.

Gross G diagnostic AUC .702354 on619/124 positives and conditional H AUC
.681089 on124/38 positives refer to different targets/populations. They are
not evidence that net ranking improved. Training labels include repeated
seconds while evaluation uses first observations; raw training transition
rates cannot be compared directly to first-observation evaluation rates.
Bootstrap intervals condition on frozen fits and exclude fitting uncertainty.

## Research consequence

Post-result descriptive support audit (`results/request314-support-audit.json`),
with no refit or selection: original97 first saved training states have
30 actual-gross5,26 whole-net5 and9 stopped-net5 cases. Common619 evaluation
first states have124/81/38. Day/episode-weighted gross prevalence is30.87%
versus20.22%; weighted stopped-net/gross conversion28.89% versus30.79%.
Thus the apparent all-state raw training40.0% versus evaluation30.6%
conversion gap disappears when the descriptive training clock is restricted
to first saved states. This does not establish causal stationarity: historical
training was selected, broad evaluation was hash sampled, days differ, and the
training sample is small. Known preentry cost-drag median1.147% versus1.109%
does not explain the different stage counts by itself. This comparison argues
for matching acquisition and observation support before adding head complexity,
not for fitting a new rule on these evaluated groups.

R312 temporal features, R313 blanket evidence waiting and R314 factorization
do not establish stronger net-opportunity ranking. Keep original M as a
development reference, with no trading-policy promotion or threshold tuning.
The decomposition identifies material execution-cost and stop-path losses;
it does not justify weakening either rule. Next research should separately
preregister a matched-clock, same-risk-policy audit of causal trajectory and
entry repricing, using chronological crossfit before testing a WAIT/ENTER
controller. No new timing rule is fitted to these outcome groups. June HOLD
and July–August remain sealed.

## Official numerical reproduction

All1,230 JSON numeric fields reproduce within1e-9 (maximum2.2661e-10),
with exact nonnumeric values and gate flags. Saved828 scored identities keep
identical probability ranks, features, stage/final labels, missingness and
identity metadata. Maximum H probability error7.0870e-10 and product P error
3.6431e-10; R/D errors1.11e-16. All4,526 training-stage rows are numerically
identical. Official original-M replay error2.22e-16. This is numerical
reproduction, not byte-identical JSON. The exact official JSON is saved as
`results/request314.json`; local JSON and detailed reproduction differences
remain separately marked. Stage transitions, same-target rank results and
failed gates are unchanged. No subsequent experiment is dispatched by the
results-only publication.
