# R326: sequence loss improves the failed rate baseline; extra predictive memory is unproved

R326's rich prediction-only sequence head F improves the matched adjacent-likelihood
head M on whole-episode timing, local timing and Brier. However, F does not establish
extra timing value over S, the identical-feature memory-free probability classifier.
Admission/calibration checks also fail. Local result:15/32 preregistered checks.
This is a useful objective-contract result, not an adoptable model or proof of
state evolution. The repaired official replay exactly reproduces every numeric result and score rank.

## Fixed population and supervision

Only strictly out-of-time May6–8 records train the new heads:7806 observed states,
266 observed episodes,7327 complete targets,256 complete-target episodes/weight
mass,51 positive episodes. Adjacent support7066 transitions/218 episodes includes
77 onset rows/33 episodes and85 decay rows/41 episodes over3 dates. Conditional
M mass214+45=259; sequence mass256; S later-target mass190.150914, with first loss
constant. These supervision differences are part of the registered contract;
conditional likelihood is not inherently a bug and the result has no sole-cause
teacher-forcing attribution.

All feature arms fit preprocessing on the same7066 current-transition states with
independent mass218; F/M/S share the exact rich transform. New heads and controls
share each episode's canonical frozen G first probability. Training G uses strictly
preceding dates, replaying each saved fold. Every observed clock, including censored
outcomes, evolves the prediction; only complete targets enter loss.

Evaluation retains19530 observations,18793 complete states,655 complete episodes,
828 original identities,78 mixed timing episodes and70 local timing episodes with
all8623 original eligible pairs. First/clock/labels/censoring/costs/stops are preserved.
The May development period is reused; no sealed dates or raw market requests.

## Development metrics

Admission means same-day between-original-episode AUROC; whole timing is within-
episode AUROC; local timing is the inherited30s pair rank. Brier is lower better.
T38 already contains causal history and lagged deltas: S is memory-free in its
probability recursion, not a model stripped of all trajectory-derived features.

| Arm | Admission | Whole timing | Local timing | Brier |
|---|---:|---:|---:|---:|
| F: rich sequence NLL | .636132002 | .521605103 | .570513073 | .052285312 |
| H: context/history sequence | .629907004 | .505781489 | .459299270 | .051929797 |
| A: age/history sequence | .630409548 | .529254018 | .501020659 | .052706969 |
| K: constant sequence rates | .630864462 | .529254018 | .501020659 | .052709524 |
| M: matched adjacent NLL | .619602500 | .394331216 | .440015011 | .056556230 |
| S: rich stationary logistic | .631164747 | .520428132 | .556653548 | .052189945 |
| B: current frozen G | .618618247 | .532915297 | .484978269 | .051957113 |
| B0: own-first G | .639103990 | .500000000 | .500000000 | .054755010 |
| P: frozen R325 rich rate | .607212535 | .368617340 | .422886203 | .055368078 |

Frozen W remains whole .671629920/local .599223212, descriptive raw-score evidence
with weak admission, not a calibrated probability substitute.

| Contrast and metric | Point difference | Day95% CI | Ticker-day95% CI |
|---|---:|---|---|
| F−M whole | +.127273887 | [.061403,.200991] | [.063963,.184345] |
| F−M local | +.130498062 | [.039761,.215173] | [.037145,.227469] |
| F−M Brier | −.004270918 | [−.005417,−.003249] | [−.005774,−.002514] |
| F−S whole | +.001176971 | [−.017832,.020231] | [−.016111,.019109] |
| F−S local | +.013859525 | [−.028761,.054734] | [−.025549,.053378] |
| F−S admission | +.004967255 | [.000068,.014418] | [−.000579,.012039] |
| F−B admission | +.017513755 | [−.038527,.056187] | [−.008219,.044358] |
| F−B whole | −.011310194 | [−.077905,.060103] | [−.083740,.056661] |
| F−B local | +.085534804 | [.024849,.145098] | [.005142,.168291] |
| F−B Brier | +.000328199 | [−.000306,.001242] | [−.000341,.001027] |

Both1000-draw schemes retain original independent weights and fixed fitted models.
They omit parameter/initial-model fitting uncertainty. F beats B admission on6/8
dates and Brier on6/8 dates, but its aggregate Brier is worse and first-only B0
admission is higher. Whole timing also trails A/K/B. Passing local timing against B
and M does not rescue17 failed checks; all failures remain in results JSON.

## How much memory remains?

Complete nonfirst observations use their ORIGINAL full-complete loss weights,
without renormalizing after first removal. Their18143 rows carry489.866172 mass.
F's retention z=exp(−(a+b)Δt) has weighted mean .181185179, median .019571264,
10th percentile4.60e−17 and90th .641160798. H mean .054695589/median4.16e−6;
A/K retain mean .963080166/.963943065. M mean .517365371.

Thus rich F usually forgets most prior predictive probability by the next clock.
This is a diagnostic of the fitted approximation, not a probability of a real
market regime switch or proof that all useful trajectories disappear. On gaps>30s,
F and S whole timing are both .287634759; they are poor despite strong aggregate
between-episode ordering. Uncertain F−S whole/local intervals and existing T38
history make a new memory-benefit claim unsupported.

## Chronological check is materially weaker

May6 has no earlier out-of-time meta day. All4602 observations/100 first clocks
remain; all six new heads have4502 unsupported later scores. Full7806 diagnostic
rows remain, with3304 common finite rows, including May6 firsts. May7 trains only
May6; May8 trains only May6–7. No fake initial-fit rates or date omission.

| Fitted May7/8 only | Admission | Whole timing | Local timing | Brier |
|---|---:|---:|---:|---:|
| F | .677749923 | .593727278 | .539933594 | .089198184 |
| M | .673914809 | .534688237 | .594127957 | .090712155 |
| S | .686603469 | .596905773 | .496032968 | .089580648 |
| B | .752820012 | .556293723 | .595865701 | .085850686 |

Only25 mixed/23 local episodes support this slice. F−M local gain does not transport
to this chronological diagnostic; B remains stronger on admission, local timing
and Brier. Report this alongside the8-day result, not as fresh validation.

## Numerical validation and execution record

Critical Ruff passed. Initial suite1158 tests, then1160 after canonical-initial
regressions, passed;159 pre-existing warnings. All18 full/fitted-fold fits converge;
all15 rate fits satisfy normalized gradient∞<=1e−6 (full maximum9.83e−8).
First loss remains constant; finite-difference tests include ragged/censored
histories and extreme hazards. Pure inference ignores outcomes, preserves future
mutation/truncation causality, and never skips a censored observation.

Prereg e1e26f3d2c59b0d92b669ae5651c32e1c7ad4e19 precedes real fits. Original
implementation252b33d3486ca0c34bf2f9e546f934e58ea3569e passed full CI37623360107.
First official attempt37623558937/source c0acb233a2be0389ebd4882ba584316f972b7495
failed at first equality after all fits; partial artifact11482753886/SHA256
34f6b7d52dc08d85aaf817e325e506e5206db3a654d18547393624d98e306907 is authenticated
and preserved. It is NOT a successful reproduction and produced no result JSON.
An almost-flat A onset intercept differs by1.00e−8 between initial local/official
fits, beyond the registered coefficient tolerance; it is also not called reproduced.

Amendment24d400d884c86cae6ddb77744c92168a24e72cd2 precedes repaired fits: verify
reconstructed frozen G and saved scores within unchanged1e−9, then reuse pinned
R325 B bits for common training/evaluation current/first G. Local reconstruction
versus original saved G differs up to1.11e−16; canonical local G error is0. No
objective/weight/parameter/seed/tolerance changes, fit selection or optimizer retry.
Original local output and failed official audit remain separate. Repaired local
metrics and all32 gates are IDENTICAL to the original local run. Repaired official
source41117f2efb3b1dc9eef8a81c1ee396f6a64991e9/run37624854775 succeeds. Artifact
11483342814/SHA25658c9105bd2b8d1440426563423f66c9d4508923a0d40a62a36fc418b28d7d829
has authenticated source/digest. Main JSON5462 and fit-audit1227 numeric fields have
maximum error0; main JSON bytes are identical. All seven parquet tables have
maximum numeric error0, exact original identities/labels/missingness and tie-aware
score ranks. No tolerance or tie rule was relaxed. Both official attempts are
recorded; the one-shot workflow is retired after verified replay. The executed
workflow remains recoverable from that source commit. Full audits/results/provenance
are in research/results/request326*.json; Draft PR99 remains unmerged.

## Next experiment

R327 will test **observed feature-path innovation**, because merely carrying the
last fitted probability adds little beyond a same-feature snapshot in R326. Add a
fixed30s causal EMA innovation of the seven already selected momentum coordinates;
compare a matched first-anchor innovation and a history-feature snapshot classifier.
This distinguishes evolving observed history from episode anchoring and from
probability recursion. Details/support are in request327-proposal.md and
request327-training-support.json. No R327 fitting or evaluation has run. Successful
reproduction is a prerequisite; retain the weak admission/calibration requirements.
