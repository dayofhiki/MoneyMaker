# R325 official findings: elapsed-time consistency does not repair timing

Primary rich rate model F passes9/24 registered checks and fails15/24. Its whole
timing .368617 is almost unchanged from R324 .367627, below frozen current G/B
.532915. Admission .607213 versus B .618618, local30s .422886 versus .484978,
and Brier .055368 versus .051957 do not support adoption. Retain the existing
baseline and every failed gate; do not select a secondary model as fallback.

R325 satisfies the tested continuous-time composition/limits and all fits
converge. Its failure therefore is not evidence that the implementation forgot
elapsed seconds or that optimizer convergence failed. Time-consistent rates
alone, in this tested model/likelihood contract, did not recover conditional
timing. Parameterization, joint likelihood and removing the gap covariate also
change together, so this is not an isolated causal test of clock spacing.

## Registered scope and execution

Remote registration d282aa234adcf50b7bc456485f14007c23efee22 precedes every
real R325 fit/evaluation. Parent R324207a51330b5dce98ba2e7ae1257a198e5355cf91,
Draft PR97. Implementation fbfb66bcc76e8655a8fed79fdaba5dddcb1e86de passes
full CI37618460314. Official source9dae06eda16def3b132d2483f62ed87fe8fdfaa4,
cached run37618872764/artifact11481140668 succeed. ZIP SHA256:
9e1608eaa71a13f149918c176a85a97ccf82289289394b4be5921bcc55dafdaa.

All12 pinned inputs verify. A first local invocation stopped at a missing cache
link during input checks BEFORE any fit; repairing the link and verifying all
hashes enabled the sole real local experiment. No optimizer/hyperparameter/model
retry or new data acquisition occurred. The official cached experiment ran once.

Training remains9325 observed states/347 observed episodes,8792 complete states
in336 complete episodes. Original complete adjacent transitions are exactly the
R324 manifest:8445 rows/286 episodes, onset38 and decay48 independent episodes
across4 dates. Transform weight mass286; conditional supervision279/54, joint
mass333. Both log-rate heads use one shared transform per representation; K has
two free intercepts and no slopes. First gap and q heads remain missing.

Every828 evaluation identity and19530 observation remains,18793 complete states
in655 complete episodes,78 mixed whole-timing episodes and70 local episodes with
all8623 unchanged pairs. All679 observed first predictions are identical across
seven arms, including650 complete first labels; no first-clock improvement.
Every7806 chronological held observation remains and is scored; models and
initial G fits use strictly preceding original dates. No label/censor column
enters inference, and censored observations remain in every recursive history.

## Official results

| Arm | Same-day between-episode AUROC | Whole timing AUROC | Local30s rank | Brier |
|---|---:|---:|---:|---:|
| F rich T38 rates, primary | .607213 | .368617 | .422886 | .055368 |
| H context/history G12 rates | .618549 | .493097 | .463284 | .052835 |
| A three HISTORY rates | .507965 | .484174 | .491513 | .053041 |
| K constant rates | .572595 | .500864 | .524780 | .052863 |
| B frozen current G | .618618 | .532915 | .484978 | .051957 |
| B0 frozen G first on same later labels | .639104 | .500000 | .500000 | .054755 |
| P frozen official R324 rich recursion | .614576 | .367627 | .433274 | .054543 |
| W unchanged pair-only reference | .467818 | .671630 | .599223 | not a probability |
| Common G training prior | — | — | — | .050811 |

F pooled AUROC .602401/AP .076414 versus B .613499/.076236. The tiny AP point
difference does not override the failed primary metrics. All seven probability
arms lose to the common prior on Brier. F admission improves on3/8 dates versus
B; F Brier is worse on all8. Training-label support is sufficient for the
registered checks but is not thousands of independent transition examples.

| F-B difference | Point | Day95% interval | Ticker-day95% interval |
|---|---:|---|---|
| Admission AUROC | -.011406 | [-.071524,.049837] | [-.058082,.043607] |
| Whole timing AUROC | -.164298 | [-.254372,-.078921] | [-.244383,-.080609] |
| Local30s rank | -.062092 | [-.204604,.083577] | [-.193419,.065882] |
| Brier (positive=worse) | +.003411 | [.002265,.004756] | [.001588,.004939] |

Both interval schemes support worse whole timing and Brier. Admission/local
differences remain uncertain. F-H admission intervals include zero; whole timing
is worse under both schemes. F-A admission intervals [.041910,.154462] and
[.046146,.152510] are positive, but F-A whole timing and Brier are worse under
both. Rich inputs add admission information beyond age in this architecture;
they do not show an increment over context/current G.

Against failed R324/P, whole timing gains only+.000991 with intervals
[-.005795,.012253] and [-.017805,.020607]. Admission changes-.007363 and local
-.010388, with both intervals including zero. Brier worsens+.000825; its day
interval [-.000033,.001802] includes zero and ticker-day [.000041,.001717]
is positive. No improvement claim over P is supported by both schemes.
Every metric has1000 valid draws; intervals condition on fixed fits and omit
initial/rate fitting uncertainty. Repeated May use is not fresh validation.

Nine passing gates: all seven support checks and both F-A admission interval
checks. Fifteen failures: four aggregate performance checks; majority dates;
both F-B admission/whole/local/Brier interval checks; both F-H admission checks.
No tested gate, initial model, features, optimizer or rate unit was changed.

Chronological diagnostic: F admission .678519 versus B .755332, whole timing
.504473 versus .630452, local .428607 versus .589252 and Brier .081877 versus
.079674. F improves some chronological timing points versus P, but remains worse
than B and context H. Fixed <=30s gaps yield F whole timing .386295 versus B
.527984; >30s yields .167842 versus .455024. These smaller-support diagnostics
do not identify a failure cause or select a new sampling interval.

## Bottleneck and next design

Both R324 and R325 optimize adjacent transitions conditional on the TRUE previous
latent training label, then deploy a recursion using an uncertain previous
prediction. Under a correctly specified conditional Markov model, such filtering
can be valid; this difference is not automatically a mathematical error. With
approximate dynamics, sparse conditional support and initialization uncertainty,
the adjacent objective does not directly minimize errors of the deployed entire
recursive trajectory. Direct sequence-loss training is the next hypothesis,
not an established explanation of these failures.

R326 will keep actual-time rates but train on its own unfolding predictions,
with loss only at complete current labels. It will use strict past-fit G initial
scores from official chronological TRAINING checkpoints, not in-sample/future-fit
initial scores. A new training-only audit finds7806 observed states/266 observed
episodes;7327 complete states/256 episodes,51 positive episodes;7066 complete
adjacent transitions/218 episodes,33 onset and41 decay episodes over3 dates.
See request326-training-support.json. No R326 rate/model/evaluation has run.

Use matched adjacent-likelihood rates on the same reduced training population
and a matched current-only stationary classifier. The latter is essential:
sequence loss could drive rates to an instantaneous classifier, yielding a gain
without useful memory. Require benefits beyond BOTH matched controls and the
frozen current G baseline. Keep initialization, original clocks/censoring, local
pairs and strict chronological diagnostics; May6 lacks an earlier out-of-time
meta-training day and must remain explicit unsupported scope. Exact proposal in
request326-proposal.md; register its final contract before fitting.

## Verification

Full local suite1139 tests/critical Ruff pass, including exact matrix exponential
agreement, constant-input composition, zero/large-gap limits, analytic gradients
over tiny/large hazards, extreme-log-rate stable loss, endpoint likelihood and L2,
conditional independence mass, causal-only inference, future/label/censor mutations,
single-class support and no-retry optimizer failure. All16 full/fold fits converge;
maximum normalized gradient infinity norm8.604901e-8, below1e-6.

Official replay and implementation full CI succeed. All3439 main JSON numeric
fields and573 fit-audit fields reproduce with maximum error0; main JSON bytes
are identical. Seven parquet tables are exactly equal, including labels, clocks,
weights, missingness and all probability/q/stationary tie-aware ranks. Executed
source and artifact ZIP digest verify. Exact official/local results and audits
are committed; the one-shot workflow is retired after verification.

Standing routine research/publication authorization applies. Main policy,
costs/stops/labels, sealed June HOLD/July-August, promotion and live trading remain
unchanged. This is a latent-target development experiment, not a realized-return
strategy result. No raw market data was acquired.
