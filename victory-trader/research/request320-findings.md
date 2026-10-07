# R320: relative learning improves conditional timing, not cross-episode selection

R319's simple lag-difference classifier did not identify the better phase within
an episode. R320 registers remote385560cf37e2bb61cd72e2f4d7137a97d45e0218
before real fits, then tests a fixed within-episode pairwise objective. Draft
PR93 is stacked on official R319/PR92. User standing research authorization
applies; no extra acquisition, policy promotion, cost/stop change or sealed dates.

## Fixed support and matched objective comparison

Use exact official R319 trajectories/transforms. Training has9,325 observations
from347 observed episodes,8,792 complete states from336 episodes. Of those,
50 episodes have both positive and negative labels and supply47,825 pairs.
The pair ledger retains all347 observed training episodes, including no-pair
episodes; it does not invent observations for originally unobserved episodes.
Pair fitting mass is50 independent episodes, not47,825 independent examples.
Preprocessing stays the exact R319 training-only transform on336 complete
episodes. These unsupervised and supervised supports are explicitly different.

Every positive-negative pair belongs to ONE episode. Each endpoint's input is
causal at its own clock. Both orientations have complementary labels; equal
day/episode weights total50. At inference score one current state, without any
future episode average, later endpoint or outcome. C=.1, no intercept, fixed
linear learner and unchanged regularization/lag/window; no search.

U uses Q9+history support, V uses M28+support, W adds the seven fixed30s lag
differences. Z is a matched STATE classifier on exactly the same50 mixed
episodes/2,534 endpoint states, with the exact W transform and learner settings.
Endpoint marginal weights give25 positive and25 negative mass, sharing the same
episode/day support and total50 mass as W. W-Z isolates the relative pair loss
from state classification more cleanly than W versus the old pooled classifier.
The loss/design vectors still differ by definition; it is not a causal market
treatment. The old R319 S/T probabilities are copied only as rank references.

All828 evaluation identities,19,530 causal observations and18,793 complete
states/655 episodes remain. Timing AUROC is CONDITIONAL on78 episodes with both
classes. It measures relative good/bad-state ordering there, not the ability to
know which of the655 episodes will offer a viable trade. Every original input,
label, missingness flag and score remains unchanged. Raw U/V/W/Z values are
relative scores, not calibrated absolute probabilities; no Brier or profit claim.

## Fixed development results

| Head | Within-episode timing AUROC | Between-episode same-day AUROC | Pooled AP |
|---|---:|---:|---:|
| U pairwise controls + support |.575280|.445428|.045496|
| V pairwise M + support |.696270|.481387|.062138|
| W pairwise M + support + lag differences |.671630|.467818|.055463|
| Z matched state classifier, W inputs |.627509|.515772|.072340|
| S original pooled classifier, V inputs |.489695|.591274|.066376|
| T original pooled classifier, W inputs |.480807|.590160|.065378|
| W0 own first score held constant |.500000|.505355|.068698|

W-T timing difference+.190823 has day95% CI[+.132585,+.246824] and ticker-day
CI[+.108597,+.275152]. This total contrast also changes supervised support,
balance, effective weight/intercept, and cannot attribute the whole gain to loss.

The matched W-Z timing increment is+.044121, day CI[+.014991,+.080835] and
ticker-day CI[+.010027,+.080320]. Thus changing to the pair-relative objective
adds conditional timing information even after matching the declared support,
transform, class balance, total fitting mass and intercept. This is narrower
evidence than establishing an executable or generally superior trader.

W-U timing increment+.096350 has positive intervals under both schemes:
day[+.051370,+.156758], ticker-day[+.032360,+.157401]. The whole directional
input package helps within-episode ordering under this fixed objective; this
does not isolate any individual feature or prove the lag additions help.

W-V timing difference-.024640 is inconclusive: day CI[-.061652,+.006609],
ticker-day[-.069759,+.026282]. W improves on V on only three/eight dates.
V-S+.206575 has positive intervals under both schemes, but V is secondary;
do not pick V as a fallback because its point timing rank is highest. There is
no evidence requiring the additional seven lag differences over the snapshot
package, and no registered model is promoted from these known May outcomes.

W-W0+.171630 has positive day[+.128390,+.213161] and ticker-day
[+.101715,+.236908] timing intervals. The constant-first control confirms a
within-episode distinction, not good cross-episode admission. W's same-day
BETWEEN-episode ranking is only.467818 and pooled AP.055463. Combining conditional
timing and choosing which ticker is worth trading remains unresolved.

Nine/thirteen fixed checks pass. Failed: W does not beat V/U/T/Z collectively,
does not beat V on a majority of dates, and neither W-V timing interval has a
positive lower limit. Every draw for the primary timing contrasts is valid
1,000/1,000 under both schemes. All gate failures remain; no fallback, new
threshold or policy follows this study.

## Temporal diagnostics and remaining interpretation limits

Chronological fits use8/25/44 preceding mixed episodes for May6/7/8. There is
no no-pair fold. Held/later-day labels/transforms stay excluded. Timing rank is
W .670906,V .658217,Z .648905,T .553430. This is different support from the78
final development episodes and not independent confirmation or grounds to
choose W over V after viewing both results.

Fixed descriptive later-phase W timing is.621838 in30-120s and.621659 in120-300s,
versus T .493431/.457558. Lag-available W timing is.641521, below V .674939.
Phase reports condition on different mixed episodes and reweight their subsets;
they are not a waiting-delay or entry-rule selection and have no new significance
test. The first30s has no available lag and cannot validate lag features.

The fitted W coefficients include negative elapsed-time/observation-count terms.
Coefficient magnitudes are descriptive, not causal importance. Some global timing
gain may reflect remaining-horizon/episode-age decline, rather than discriminating
nearby evolving price/activity states. W-U controls for support/activity as a
package, but an explicit age-only and time-proximity diagnostic would narrow this
interpretation further. Treat this as the next question, not a cause proven by
coefficient inspection. request321-proposal.md sets that follow-up question.

The main bottleneck has narrowed: pooled classification was insufficient for
within-episode timing; a relative objective can help. Absolute opportunity
admission, age effects, conditional support limits and sequential execution still
separate this finding from a profitable continuous trader. Do not search deeper
static trees or more lag windows from these known outcomes.

## Official execution and verification

[Official run37590161637](https://github.com/dayofhiki/MoneyMaker/actions/runs/37590161637)
**succeeded**, source9773ed5d1af0081436d82d54fd6e4974617add1c,
artifact11467839297. ZIP digest verifies:
cd24be6d1470c52d9b0e83c5671746e28689e3603939b71d9fb592ee2a4e21aa.
All4,419 JSON numeric fields and nonnumeric values reproduce exactly; JSON bytes
match. All four parquet checkpoints (47,825 pairs,347 ledger episodes,19,530
evaluation states,7,806 chronological states) match all values, identity/clocks,
labels, missingness and full relative-score/probability pair ordering exactly.
Maximum numeric error0. Exact official/local JSON and verification are committed.

Full local suite: **1,075 tests passed**; critical Ruff passes. Both official full
source CI workflows succeeded, and the official cached run's tests/Ruff succeeded.
R319 rank baselines replay without error; frozen state transforms verify.
One-shot workflow is retired after closeout, so documentation updates do not
repeat experiments. Main/research-request.json, event tracks, June HOLD and
July-August remain unchanged. PR93 stays draft; no policy adoption or merge.
