# R319: later opportunities exist, but fixed lag differences do not identify them

R318 official run37476099213/PR91 confirms static joint interpretation did not
repair the original first-clock ranking. R319 tests continued evaluation AND one
fixed trajectory package, registered locally in commitd61073a before clocks,
labels and new fits. Implementation commitbc5fa3e precedes closeout. The initial branch publication was blocked by automatic approval review. The
user explicitly authorized publication/cached execution on2026-10-07 and granted
standing authorization for routine research of the same scope. Draft PR92 is
now published; official cached run37588633963 succeeded. No merge, market request,
policy promotion or sealed-date access occurred.

## What changed

Same828 original May11-20 identities; every completed aggregate in the inherited
HOT+300s window.19,530 evaluation clocks/679 observable episodes are retained.
18,793 states from655 episodes have complete labels;737 stay censored and149
episodes have no observation. Labels remain next-open entry, BASE costs,
-2.5% absorbing stop and HOT+1h/session cap. These are alternative cash decisions,
not sequential trades. All original first inputs, labels and missingness replay.
C/Q/X/K first probabilities match R318 with maximum error4.16e-17.

History is built on all9,325 training observations BEFORE censor filtering.
All new fits use the identical8,792 complete states/336 episodes and original
independent weights. S adds elapsed/count/lag-age support to M28; T adds seven
current-minus-lag30s directional/activity differences; G removes M's direction
package but retains support. Linear regularization and transforms are unchanged.
C/Q and passive X/K replay their original pipelines. Each own-first score is
also held constant across the SAME later state labels as a longitudinal control.

## Opportunity support versus ability to identify the good phase

First-label +5% reachable opportunities appear in38 episodes; across continued
observation they appear in89, including51 not positive at the first clock.
This is outcome-described support, not a deployable rule for selecting those51.
Only five previously censored first episodes gain any complete later label.
Thus later support is not predominantly a new complete-episode population.

The main comparison gives every day equal mass and every episode equal mass
within day. Correlated seconds share their episode's weight. SAME-day rank
excludes pairs whose endpoints belong to the same original episode. A separate
within-episode rank is conditional on78 episodes containing BOTH label classes;
constant-first scores have exactly.5 there.

| Fixed head | Between-episode same-day AUROC | Within-episode AUROC | AP | Brier |
|---|---:|---:|---:|---:|
| C current snapshot linear M |.598489|.463012|.069936|.052675|
| Q current snapshot controls |.628529|.513330|.087847|.051344|
| X passive joint snapshot M |.643225|.468583|.085590|.052909|
| K passive joint controls |.633016|.486107|.083523|.052895|
| S snapshot M + history support |.591274|.489695|.066376|.053036|
| G controls + history support |.618618|.532915|.076236|.051957|
| T support + fixed lag differences |.590160|.480807|.065378|.053191|
| T0 own first score held constant |.616677|.500000|.073854|.054357|
| C0 own first score held constant |.619948|.500000|.076432|.052749|
| X0 own first score held constant |.647708|.500000|.090567|.052784|

Own-prior Brier is.050811: every dynamic head is worse than its prior. The
passive X point rank is not a selected winner: X-X0 is-.004482 with both
cluster intervals crossing zero, and its within-episode rank is below.5.
Do not compare X's all-state.643225 to R318's first-clock.603439 as an isolated
improvement; different state labels and weights enter those two numbers.

T-S rank difference is-.001114: day CI[-.009209,+.004365], ticker-day
CI[-.007443,+.005241]. T-S within-episode difference-.008887 also has both
intervals crossing zero. The fixed trajectory package establishes no increment.

T-G rank difference-.028458 has ticker-day CI[-.057340,-.001210], but day
CI[-.066526,+.005052] crosses zero. Do not describe rank harm as established
under both schemes. T-G Brier difference+.001233 is positive under both schemes,
as is T-C Brier difference+.000516. Direction/history does not establish a benefit.

T-T0 rank difference-.026517 has day CI[-.050412,-.001280], while ticker-day
CI[-.055015,+.000052] narrowly crosses zero. C-C0-.021459 has day
CI[-.039681,-.002113] and ticker-day[-.045256,+.000353]. Updating probabilities
does not automatically improve opportunity ranking. T-T0 Brier improves under
both schemes, but T still loses to the constant training-prior Brier. Better
probability error relative to a particular first score is not better timing.

Only2/14 preregistered checks pass: positive-episode and positive-date support.
T-S improves on four/eight dates, not a majority. All primary rank draws are
valid1,000/1,000 under both day/ticker-day bootstrap schemes. Intervals condition
on the fixed fits and omit fitting uncertainty.

## Phase and chronological checks

The lag-available subset contains15,945 complete states from545 episodes. There
T same-day rank.650910 remains below S.653683/C.662123/G.662324, and T
within-episode rank.450190 remains below S.486053. These are descriptive fixed
partitions; their metrics reweight their own subsets and cannot be added together
to reconstruct the whole-window metric. They are not an entry-window selection.

The first30s descriptive T within-episode.693755 uses only18 mixed episodes and
contains NO available30s lag at all. It cannot support lag trajectory improvement
or a new30s trading rule. Model coefficients/transforms also differ between T/S.

Preceding-day chronological training diagnostics exclude held/later-day labels
and transforms. T same-day rank.731936 is below S.735068/C.742382/G.755332.
T within-episode.553430 is below S.559233/G.630452. Dates/support differ from
the final development comparison; these diagnostics are not independent proof.

## Narrowed bottleneck and next question

Continuing observation exposes additional opportunity phases, but current-state
interpretation and this one simple30s lag-difference representation do not learn
to rank the good phase within an episode. This does NOT refute trajectory models
in general or establish a recurrent architecture is required. Sparse observation,
remaining upside relative to the current entry price, target design, limited
within-episode support and model capacity remain possible explanations.

A useful next isolated question is **training objective**, rather than deeper
static interactions or another lag/window sweep: train to compare good and bad
states WITHIN the same episode, so ticker/day background cannot solve the task.
R319 has50 mixed-label training episodes/47,825 positive-negative state pairs;
the effective support is50 episodes, not47,825 independent samples. The proposed
R320 protocol is inrequest320-proposal.md and has not been executed. Keep the
snapshot versus trajectory input ablation and distinguish timing ranks from
absolute probability calibration. No model/policy promotion follows R319.

## Validation and reviewable publication package

Full local suite: **1,063 tests passed**. Critical Ruff passes on allsrc/tests
and the verification script. A second unchanged complete offline execution
reproduces all6,027 JSON numeric fields with maximum error0; JSON bytes match.
All six parquet checkpoints match identities, clocks, labels, missingness and
all probability pair ordering exactly. Numeric maximum error0 on each table.
request319-reproduction.json records this LOCAL-to-LOCAL verification; no claim
of official reproduction is made. R311 raw/cohort hashes match the existing
official diagnostic, and R317/R318 downloaded ZIP digests verify.

## Official closeout

[Official run37588633963](https://github.com/dayofhiki/MoneyMaker/actions/runs/37588633963)
succeeded, source8084ff22b535ae76e8adf57b0485f6bbe34ec2c6, artifact11467966003.
ZIP SHA256 verifies:23a01bac33e992b35e2dc6fbfaa8eb469cd4119fd2cf006047cb09b75df06cd2.
All6,027 JSON numeric fields reproduce within1e-9; maximum error7.48e-11.
JSON bytes differ and are not claimed identical. All six parquet checkpoints
retain identical identities, clocks, labels, missingness and probability pair
ordering; maximum frame numeric error9.33e-11. Both full source CI workflows
succeeded. Exact official JSON and official-to-local audit are committed.

The branch-only one-shot workflow is retired after verified closeout, so findings
updates do not repeat the experiment. Draft PR92 remains unmerged. Main,
research-request.json and the event tracks stay unchanged; no policy adoption.
