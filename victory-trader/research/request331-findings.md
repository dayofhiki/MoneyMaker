# R331: later observations recover independent targets, but execution observability fails

**9/10 registered checks pass; the feasibility audit fails its 90% resolution
requirement. No model, policy or profitable account is established.** Extended
observation clears the minimum independent-target floor, but cannot repair the
economic-label/execution bottleneck under the inherited next-print contract.

Remote preregistration d61395380d6ad49ff17d0506985d7f1d9e620c5b precedes all new
clocks and outcomes. Evaluated source175645954cf2216781992a4619356c49b348332b
contains the previously validated implementation. All407 original broad May5–8
and828 May11–20 identities remain. Primary feasibility is305 May6–8 identities.
No fit, new market request, event routing or sealed-data opening occurred.

## What improved, on the same support definition

All original completed-print early clocks remain. The later arm adds only the
first completed print in each fixed30s bucket after5min through the inherited
HOT+60min/session terminal. All phases use the exact R33060s hold, BASE−2.5%
completed-close stop,3s entry/exit bounds and original cost scenarios.

| May6–8 primary scope | EARLY | ADDITIONAL_LATE | EXTENDED union |
|---|---:|---:|---:|
| Observed clocks |7,806|13,634|21,440|
| Observed original episodes |266|297|300|
| Episodes with any resolved immediate target |114|159|170|
| Episodes with any cost-compatible resolved target |105|146|156|
| Episodes with positive compatible BASE target |38|41|59|
| Compatible resolved clocks / compatible clocks |4,209/7,457|2,788/12,629|6,997/20,086|
| Pooled compatible row resolution |56.4436%|22.0762%|34.8352%|
| Complete compatible original observation-weight mass |36.218310|30.394971|33.715903|

The union gains51 compatible complete episodes and21 positive episodes; no early
support is lost. Each of the3 dates gains complete/positive support: May6 complete
47→63,positive18→23; May7 complete29→50,positive13→22; May8 complete29→43,positive7→14.
Both payoff signs exist on all3 dates. The registered>=150/30 floors therefore
pass (156/59). They are feasibility minima, not evidence of a sufficient learner
or untouched validation. R330's104 common ENTER/WAIT episodes are a different
estimand; the matched R331 early comparison is105 compatible immediate episodes.

The equal-day/equal-identity complete-episode support increment has day95% CI
[.140000,.216495] and ticker-day[.124115,.211759]; positive support increment has
day[.046296,.092784],ticker-day[.040138,.099790]. These are proportions including
all305 identities, fixed-census intervals with only3 dates, not new independent
validation. Positive support is hindsight observability, never a deployable oracle.

## What still fails

Pooled union resolution is only34.8352% versus registered90%. Of20,086 causally
cost-compatible clocks,9,812 have late/missing entry references and3,277 have
late/missing exits;6,997 resolve. Later clocks have only22.0762% resolution.
None of these unknowns is filled forward, settled at an earlier print, or
converted to a loss/cash label. No best-return phase or looser lag bound is selected.

More nominal episodes do not create more complete observational weight. Union
complete compatible mass33.715903 is below early36.218310 because the original
1/(ALL observations in that episode/arm) denominator now includes sparse later
observations. Known labels are never renormalized to hide this dilution.

The distinct equal-day/equal-identity observation-mass resolution estimator is
12.4705%, with dayCI[6.9625%,22.0098%],ticker-day[10.0305%,15.2610%]. It is NOT the
pooled34.8352% gate estimator; no confidence interval from one replaces the other.
This difference further exposes how dense resolved rows can overstate coverage
of a typical original identity.

All407 training identities give compatible support131→197 and positives44→70;
union pooled resolution31.2822%. All828 reused evaluation identities give
compatible support238→339 and positives64→115; union resolution33.8591%.
Ten training and21 evaluation identities remain wholly unobserved; primary
May6–8 has5. They remain in identity ledgers, not silently removed.

All full observed-target means remain unknown because of censoring. Conditional
known compatible, original-weighted BASE mean is−1.007068% early and−1.060005%
union; union STRESS−3.056019%. These are target-distribution diagnostics, not a
selected policy/account return. Their negativity does not prove selective edge
is impossible, and positive episodes do not prove a causal learner can find it.

## Why another memory/value fit is not the next step

The post-result diagnostic uses only registered output fields, without new
targets/fits/threshold selection. Most unknown references have a later print
outside the3s bound; this does not say whether a real order could have filled.
The provider documents that stock bars are generated from qualifying trades,
and may be absent without eligible trades
([official aggregate documentation](https://www.massive.com/docs/rest/stocks/aggregates/custom-bars)).
Missing bars alone therefore cannot distinguish inactivity, trade-condition
exclusion, cache loss, absent quotes or actual failed fills. That is an inference
about the limits of this dataset, not a diagnosis of each missing interval.

Historical NBBO records include bid/ask prices, displayed sizes and timestamps
([official quote documentation](https://www.massive.com/docs/rest/stocks/trades-quotes/quotes)).
They can inform a separately justified execution proxy, but do not prove actual
receipts, depth/queue priority or fill capacity. Both original R311/R315 source
versions already follow second-aggregate next_url pagination (same client blob
032604d836e381e2bfd18fbd1b673065cd076a0c); code inspection alone does not prove
every historical response/cache page was complete.

Next priority is source/execution observation, while retaining continuous
current-state decisions and keeping explicit memory low priority. R332 proposal
is a cached lineage/gap audit first, followed only by a separately registered
source or order-contract intervention if needed. Do not repeat a tradability
classifier as a profit claim or sweep a more complex learner over this censored
support. Learned HOLD/EXIT, redeployment value and allocation remain later work.

## Reproduction, operational audit and closure

One official [run37752214023](https://github.com/dayofhiki/MoneyMaker/actions/runs/37752214023)
succeeds at the evaluated source. Timestamp-only artifact11537704285 was published
before outcomes; complete artifact11538058764 contains all results. Both archive
digests/source identities are authenticated in request331-provenance.json.
Local and official JSON numbers(5,093 total) have maximum error0; both JSON byte
streams and all8 parquet tables are exact. Missingness, clocks, phases, support
flags/reasons and payoff tie-aware ranks match. No failed or duplicate official
census run, fitting retry or tolerance relaxation occurred.

Full local1,239 tests,19 new synthetic tests and critical Ruff pass; source CI
37752213870/37752218207 passes. Post-result diagnostic script also passes critical
Ruff. No new labels/fits are created by that script. One-shot workflow is retired
in the closeout commit, preventing another scientific replay on publication.

Two earlier public-branch attempts were auto-review rejected and preserved. The
user explicitly approved R331 public design/code/hash publication and cached
execution, renewing same-scope standing approval. Shell git lacked authentication,
so authorized connected-app commits published the branch. A direct artifact URL
transfer returned403; authenticated materialization succeeded and both ZIPs were
verified. These operational failures changed no scientific rule/input/result.
Draft PR104 remains stacked/unmerged. No model/policy promotion or live trades.
