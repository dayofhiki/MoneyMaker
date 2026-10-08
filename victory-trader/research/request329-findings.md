# R329 findings: useful whole-episode smoothing is insufficient for admission or action

R329 is complete on reused May development. **27/40 pass, 13 fail; accumulated
memory is not adopted.** The binding branch decision is now activated: reduce
priority of new explicit EMA/latent/sequence-memory variants and investigate
current-state continuous ENTER/WAIT policy. This does not say history has no
information; G and the current features already encode history.

## Fixed experiment and result

Preregistration9ee116d9037b8fe3ad0e7d7b30b02f478f1e093a preceded implementation,
all real fits and new evaluations. Freeze G, gate and R328 S preprocessing;
train only accumulated-evidence F and fixed first-eligible anchor N. Current C,
mask M and G B/B0 are copied official controls. All original19,530 observations,
18,793 complete labels,828 ledger identities,655 complete episodes and8,623 local
pairs remain. Full/fold training uses original weights and strict-past G; no
full-fit meta control becomes an offset. All six supported new fits converge;
maximum normalized gradient2.243177157e-8. May6 active predictions remain NaN;
all7,806 chronology rows and3,488 common finite rows remain.

| Arm | Absolute admission AUROC | Whole timing | Local30s timing | Brier |
|---|---:|---:|---:|---:|
| F | 0.617207 | 0.629572 | 0.584276 | 0.052245 |
| N | 0.583052 | 0.538204 | 0.479761 | 0.054030 |
| C | 0.617946 | 0.604439 | 0.611142 | 0.052092 |
| M | 0.624293 | 0.540548 | 0.474031 | 0.052619 |
| B | 0.618618 | 0.532915 | 0.484978 | 0.051957 |
| B0 | 0.639104 | 0.500000 | 0.500000 | 0.054755 |

F exceeds N in admission/whole/local under both bootstrap schemes. F−N whole
+.091368 and local+.104514: day intervals[.033426,.169858]/[.042829,.164506],
ticker-day[.041825,.145229]/[.050287,.162329]. It exceeds G in whole/local:
+.096657/+.099297, with both-scheme intervals positive. The fixed first-eligible
anchor N is weak; beating it is not enough to establish added memory over current
information.

**There is a real but limited increment over current C in whole timing**:
F−C+.025133, day[.001564,.056649], ticker-day[.001795,.054380]. This is consistent
with smoothing across a whole episode, not evidence of a better deployable
admission/entry controller. F−C local is−.026867: day[−.055368,−.002798],
ticker-day[−.058598,.003355]. Admission F−C is−.000739 and both intervals cross0.
Brier F−C is+.000153 (worse); day interval is positive, ticker-day includes0.
F−G admission−.001412 and Brier+.000288 also fail both robust improvement gates.
F improves G admission on a majority of dates, but the aggregate/interval gates
fail. Do not hide that point improvement or use it to bypass the failed checks.

The13 failed gates are3 aggregate admission/local/Brier,4 F−G admission/Brier
intervals,2 F−C local intervals,2 F−C admission intervals and2 F−M admission
intervals. The whole-timing aggregate passes. All9 support/integrity checks pass.
Full interval output and every failed gate remain in results/request329.json.

For8,446 original pairs with both endpoints eligible, F local=.619116 versus
C=.652617 and N=M=G=.464651. N's fixed episode residual preserves G order on
these endpoints, as designed. This diagnostic keeps inherited pair weights;
it is not a newly selected evaluation population. On strictly fitted May7/8
chronology, F local=.587676 versus C=.610528 and G=.595866; admission/Brier also
remain weaker. Those25 mixed/23 local episodes are descriptive, not fresh
validation or a full-sample matched degradation estimate.

## Verified execution and retained failures

Successful official [run37745032388](https://github.com/dayofhiki/MoneyMaker/actions/runs/37745032388),
source79c529055b7203292a318139836cce52f574b051, authenticated archives
11534843775/11535900070. Local/official main JSON1,915 numeric fields plus fit
audit408 fields have error0. Main JSON bytes and all8 parquet tables are exact;
identities, missingness, labels and tie-aware score/correction/logit ranks match.
Archive digests and source metadata are in request329-provenance.json.

First official run37744636385/source4485a344c0caecef452b1837f1d4b305706ca15a
stopped on a pre-fit full/fold memory fingerprint guard. **Zero new fits occurred.**
Its artifacts and trace are retained. No numerical path from that first source
was persisted, so its cause is unknown. The only subsequent source changes add
failure-path diagnostics and stderr capture. All original hash guards, optimizer,
inputs, objectives, seeds and scientific gates remain unchanged; the subsequent
run passes those same guards and performs the first supported official fits.
This is not a successful first attempt and not an optimizer retry/search.

A local partial temporary training parquet remains as an exact8,109,286-byte
prefix of the complete8,484,372-byte table. The complete destination has a readable
footer, matches official bytes, and is the only file verified. The orphan temporary
was never used. It is retained and its cause is unknown; do not silently delete it
or assert an unproved runtime explanation. Atomic destination writing successfully
protected the complete output. The one-shot workflow is retired at closeout.

Local full tests1,205 pass (160 existing/expected warnings), critical Ruff passes.
Implementation CI37744376521/37744383204 and final evaluated-source
CI37745032442/37745038645 pass. No raw market requests, sealed June/July-August,
main merge, live trades or model/policy promotion.

## Current bottleneck and next experiment

The bottleneck is no longer a default missing-memory explanation. Some within-
ticker timing is learnable; smoothing adds whole-episode ranking but does not
produce superior local decisions, global admission or probability error. None
of these hindsight reachability diagnostics gives an ENTER payoff or the value
of waiting. R330 will construct explicit fixed, cost/stop/fill-aware action values
and test a small current-state policy against an ENTER-only value control and
frozen G continuous/first-clock controls. It will replay simultaneous capital
constraints separately and report unresolved account outcomes honestly.

Existing R243–274 action research prevents a superficial reset: learnable
tradability did not yield profitable entries (R245 BASE−1.3155%,STRESS−3.2824% on
19 closed trades); R256's future mean is not the value of a causal WAIT policy;
R274 prohibits fallback settlement before a missing deadline. R330 uses the
broader continuous population/current signal, a fixed causal future baseline for
WAIT targets and explicit prompt fill/terminal censoring, not future-best labels.

Independent R316B [PR89](https://github.com/dayofhiki/MoneyMaker/pull/89) remains
underpowered: execution-day20 complete sets,4 positives, difference intervals
include0, simple BASE mean−1.73%. No router/specialist is merged. It needs separate
replication and execution evidence. Current-state action research does not open
or relabel this event cohort.
