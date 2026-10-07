# R324 official findings: recursive latent transitions fail the timing task

The primary rich recursive model F passes9/24 registered checks and fails15/24.
Between-episode admission AUROC .614576 does not improve on frozen current G/B
.618618. Whole-episode timing falls .532915→.367627; local30s pair rank falls
.484978→.433274. Brier worsens .051957→.054543 on all8 evaluation dates.
Do not adopt this model or substitute a secondary arm. This rejects the tested
two-head transition recursion, not every state-evolution/trajectory hypothesis.

## Provenance and fixed scope

Remote registration3d3d01a049986e5dcf5aaa77eceb8eb9689a60c7 preceded all real
R324 fits/evaluation. Parent R3236537b156400a81fd521a5351bdc83866bb247e83,
Draft PR96. Implementation26f063b06f08e799320296cbac15fd4b5fb4779f passes full
CI37601928348. Official source d96386aca6451b1103cf18e8c31e56b54c696265,
run37602021567, artifact11473023015 succeeds. Artifact ZIP SHA256:
7ef762ae136c8040f5e737331eae38665a70ff065882313f5f864264b031df50.

Eight input hashes are unchanged. Training has9325 original observations in347
observed episodes;8792 complete observations in336 complete episodes. The plan's
336 training episodes refers to COMPLETE-label support, not every observed identity.
Original adjacency produces8445 complete transitions in286 episodes. Onset0→1
has88 rows/38 episodes; decay1→0 has99/48, each across all4 training dates.
Conditional supervision has279 previous-zero and54 previous-one episodes, not
7641 and804 independent examples. Persisted transform weights sum286 and
conditional weights sum279/54; both heads share the registered current-state
transform, without rare-class balancing.

Every one of828 evaluation identities remains. There are19530 observations,
18793 complete states in655 episodes,78 mixed whole-timing episodes and70 local
episodes with all8623 inherited pairs. Inference uses every original observation,
including censored labels, and reads no outcome/censor column. Initial p is the
same frozen G first prediction for all arms. First679 observed identities include
650 complete first labels; there is no first-clock prediction improvement.
All7806 chronological May6/7/8 held observations remain, with strictly preceding
transition and initial models and zero unsupported predictions in every fold.

## Official results

| Arm | Same-day between-episode AUROC | Whole timing AUROC | Local30s rank | Brier |
|---|---:|---:|---:|---:|
| F rich T38 + actual gap recursion, primary | .614576 | .367627 | .433274 | .054543 |
| H context/history + gap recursion | .615591 | .470489 | .459018 | .052429 |
| A elapsed/history + gap recursion | .547698 | .492507 | .511760 | .052509 |
| K constant conditional-prior recursion | .589016 | .503700 | .523389 | .052377 |
| B frozen current G classifier | .618618 | .532915 | .484978 | .051957 |
| B0 frozen G first on same later labels | .639104 | .500000 | .500000 | .054755 |
| W unchanged pair-only reference | .467818 | .671630 | .599223 | not a probability |
| Common frozen G training prior | — | — | — | .050811 |

F pooled AUROC .610742/AP .085654 versus B .613499/.076236. AP gain does not
override failed admission/timing/calibration criteria. All six probability arms
lose to the common prior on Brier. F improves admission on only3/8 dates versus B.

| F-B difference | Point | Day95% interval | Ticker-day95% interval |
|---|---:|---|---|
| Admission AUROC | -.004043 | [-.064855,.052312] | [-.054084,.050232] |
| Whole timing AUROC | -.165289 | [-.261711,-.077477] | [-.249822,-.086118] |
| Local30s rank | -.051704 | [-.201849,.114575] | [-.178113,.076515] |
| Brier (positive=worse) | +.002586 | [.001303,.003985] | [.000740,.004236] |

Whole-timing loss and larger Brier are supported by both fixed-fit interval
schemes. Local and admission differences are uncertain. F-H admission intervals
include zero; F-H whole-timing differences are negative under both schemes.
F-A admission has positive intervals [.000324,.137016] and [.013846,.121024],
but whole timing is worse under both. This supports information beyond age for
admission within this architecture, not a benefit of rich transitions over context
or over the unchanged absolute baseline. Every interval has1000 valid draws and
omits initial/transition fitting uncertainty.

Nine passing gates: seven support checks and the two positive F-A admission
intervals. Fifteen failures: four aggregate performance checks; majority dates;
both F-B admission, whole-timing, local and Brier interval checks; both F-H
admission interval checks. No gate, feature, solver or initial model was changed.

Chronological diagnostic agrees with the failure: F admission .690964 versus
B .755332, whole timing .476749 versus .630452, local .366534 versus .589252,
Brier .081647 versus .079674. The fixed >30s-gap subset also has weak F timing
.156978 versus B .455024, while <=30s is .387657 versus .527984. These overlapping,
smaller-support diagnostics do not establish the cause or select a new interval.

## Interpretation and next experiment

Recursion alone did not recover W's conditional timing signal. R324 learns two
sampled-observation transition probabilities and propagates the previous belief.
Its gap is an ordinary feature; the model does not enforce consistency between
one long elapsed interval and several shorter intervals with constant inputs.
The constant K update retains .583909 of an initial-belief perturbation per
observation regardless of elapsed time. The rich F model retains a variable
amount. This mathematical structure suggests a clock-spacing hypothesis; it is
not evidence that spacing caused the observed failure. Latent reachable-profit
labels also need not obey a two-state Markov model.

The next TRAINING-ONLY audit records actual gaps: median2s,90th percentile18s,
99th percentile83.56s, maximum240s;449 transitions/216 episodes have gaps>30s.
Those contain only10 onset and9 decay rows. No R325 rates/model or evaluation
was fitted. See request325-training-support.json and request325-proposal.md for
a continuous-time transition-rate experiment that uses actual elapsed seconds
and an explicit constant-input composition check. Keep matched initial,
context/age/constant/current controls, all-row inference and every failed gate.

## Verification

Full local suite1116 tests and critical Ruff pass. Synthetic tests cover real
gaps, no censor bridging in training, shared transforms/independent conditional
mass, prediction-only recursion without outcome columns, future/label/censor
mutations, unsupported/single-class support, initial chronology and score replay.
Official replay and implementation full CI succeed. All3291 JSON numeric fields
reproduce with maximum error0, and JSON bytes are identical. All seven parquet
tables reproduce exactly, including identities, labels, missingness, transition
weights and every probability/q-head tie-aware ordering. ZIP digest and executed
source verify. The one-shot workflow is retired after verification.

Standing routine research/publication authorization applies. This remains reused
May development with hindsight research targets, not realized returns or fresh
validation. Main configuration, costs/stops/labels, sealed June HOLD/July-August,
promotion and live trading remain unchanged; no market data was acquired.
