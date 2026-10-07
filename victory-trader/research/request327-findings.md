# R327: extra observed-path innovation does not establish model improvement

R327 F passed **9/36** preregistered development checks. Adding a causal30s EMA
innovation block to the rich sequence model did not establish timing gains over
current-feature C, fixed-anchor N or same-feature stationary S. Probability error
was worse than frozen G under both bootstrap schemes. Official replay reproduces
every result exactly. No head is adopted or promoted; all27 failures remain.

## Fixed experiment and population

Only strictly out-of-time May6–8 trains new F/N/S:7,806 observed states,7,327
complete targets/256 original weight mass. Transform support remains7,066 original
transition currents/218 episodes. T38 already includes causal history/lag deltas.
Seven extra innovations use τ30s, emit before updating, and each feature's own
last finite clock. Missing inputs carry past feature state/time and emit missing
innovations. Censored outcomes do not alter feature paths. All original clocks
advance probability; every initial probability is the pinned canonical G bit.

F is T38+EMA innovation with sequence likelihood; N replaces EMA by the first
finite anchor; S uses F's exact inputs in a stationary logistic probability head.
All current preprocessing is frozen from the corresponding R326 fit; only the
separate innovation block is new. C/D are pinned R326 sequence/stationary paths,
B/B0 current/own-first frozen G. No control refits, feature/window/τ search or
selected fitting retries. F/N have90 processed coordinates and182 rate parameters.

Evaluation retains19,530 observations,18,793 complete targets/655 episodes,828
original identities,78 mixed timing episodes,70 local episodes and all8,623 pairs
over the same8 May dates. Costs/stops, clocks, labels, censoring and sealed dates
are unchanged. Hindsight reachable net>=5 targets are not realized trading profit.

## Full development comparison

Admission is same-day between-original-episode AUROC; whole/local timing describe
within-episode ordering. Brier is lower better. Gates evaluate all original rows,
not only rows with complete new history.

| Arm | Admission | Whole timing | Local timing | Brier |
|---|---:|---:|---:|---:|
| F: EMA sequence | 0.628114491 | 0.536144420 | 0.562560438 | 0.052609929 |
| N: anchor sequence | 0.634065233 | 0.545848566 | 0.591603294 | 0.052486234 |
| S: EMA stationary | 0.623205165 | 0.529832222 | 0.544005457 | 0.052519248 |
| C: R326 rich sequence | 0.636132002 | 0.521605103 | 0.570513073 | 0.052285312 |
| D: R326 rich stationary | 0.631164747 | 0.520428132 | 0.556653548 | 0.052189945 |
| B: frozen current G | 0.618618247 | 0.532915297 | 0.484978269 | 0.051957113 |
| B0: own-first G | 0.639103990 | 0.500000000 | 0.500000000 | 0.054755010 |

| Contrast / metric | Point difference | Day95% CI | Ticker-day95% CI |
|---|---:|---|---|
| F−C whole | +0.014539317 | [-0.004331090, 0.029951899] | [-0.012675392, 0.045291274] |
| F−C local | -0.007952635 | [-0.081640928, 0.054243657] | [-0.075699126, 0.061239195] |
| F−N whole | -0.009704146 | [-0.042891741, 0.020075188] | [-0.055334775, 0.030622102] |
| F−N local | -0.029042857 | [-0.094142345, 0.021031520] | [-0.093523032, 0.019573722] |
| F−S whole | +0.006312198 | [-0.013177269, 0.029337456] | [-0.011181096, 0.024510855] |
| F−S local | +0.018554980 | [-0.003795311, 0.047426368] | [-0.002250274, 0.039621931] |
| F−B local | +0.077582169 | [0.038551554, 0.118895590] | [-0.013841132, 0.162203278] |
| F−B Brier | +0.000652816 | [0.000160877, 0.001253534] | [0.000073429, 0.001366183] |
| N−C whole | +0.024243464 | [-0.002071330, 0.050026502] | [-0.014588657, 0.065596675] |
| N−C local | +0.021090221 | [-0.016779555, 0.064632117] | [-0.016740579, 0.070558823] |

All12 added F−C/F−N/F−S whole/local interval checks fail because0 remains inside.
F−C admission falls .008017511, local falls .007952635 and Brier rises .000324617;
whole increases .014539317 with uncertain intervals. N has better point timing
than F/C, but its N−C whole/local intervals also include0 under both schemes.
The preregistration does not select N as a fallback winner.

Seven support checks pass, plus F−B admission on5/8 dates and only the day-level
F−B local interval. F improves admission over C on2/8 dates and local on4/8.
All four aggregate checks fail. F−B Brier rises .000652816, with positive day and
ticker-day intervals: a calibration deterioration, not an ambiguous gain.
Fixed-fit1,000-draw intervals preserve original independent weights and omit fit
uncertainty. Reused May development cannot establish fresh transport or profit.

## Availability is a diagnostic, not a selected validation population

The predeclared feature-availability slices show a substantial descriptive
contrast. Slice metrics recompute their own independent episode weights; episodes
overlap between slices, so these metrics are not additive.

| History availability | Complete rows | Complete episodes | Original mass | Local episodes | F whole / local | C whole / local |
|---|---:|---:|---:|---:|---|---|
| all_seven | 16980 | 460 | 332.245570 | 57 | 0.584564713 / 0.585260177 | 0.540598726 / 0.550842144 |
| partial | 1163 | 412 | 157.620602 | 15 | 0.431764590 / 0.473385231 | 0.431360696 / 0.637800645 |
| none | 650 | 650 | 165.133828 | 0 | — / — | — / — |

The partial slice has only21 mixed and15 local episodes, no slice confidence
intervals, and different episode composition. It does **not** diagnose missing
history as the causal source of failure. The all-seven slice has61 mixed/57 local
episodes; using its better point scores alone would be outcome-based selection.
None-history comprises679 original first observations/650 complete targets;
every arm is exactly G there and no local pairs exist.

Additional observed-path features also do not establish retained prediction
memory. Over18,143 complete nonfirst states with ORIGINAL mass489.866172, F's
weighted retention mean is .162969889 and median .010398895; N mean .102828756,
median .000448352; C mean .181185179, median .019571264. Rich models still largely
forget prior predicted probability. Rates are approximate fitted quantities, not
identified real regimes. On>30s gaps F/S whole timing .340241 is poor versus G
.455024; that slice has just17 mixed episodes and does not rescue the result.

## Chronological diagnosis

May6 has no preceding meta day: all4,602 rows and100 firsts remain, with4,502
unsupported later scores for F/N/S/C/D. Full chronological output retains7,806
rows/3,304 common finite rows. May7 fits only May6; May8 only May6–7.

| Arm | Admission | Whole timing | Local timing | Brier |
|---|---:|---:|---:|---:|
| F | 0.678356242 | 0.629701006 | 0.615811631 | 0.089119845 |
| N | 0.684801342 | 0.593258206 | 0.491192122 | 0.089544980 |
| S | 0.680129440 | 0.625906732 | 0.623163386 | 0.089337458 |
| C | 0.677749923 | 0.593727278 | 0.539933594 | 0.089198184 |
| D | 0.686603469 | 0.596905773 | 0.496032968 | 0.089580648 |
| B | 0.752820012 | 0.556293723 | 0.595865701 | 0.085850686 |

F improves fitted May7/8 whole/local point timing over C, but this slice has only
25 mixed/23 local episodes. It does not establish gains in the eight-day held
development evaluation. G remains much stronger in admission and Brier; S local
also exceeds F here. Keep this limited chronological check beside the full result.

## Execution, validation and authenticated reproduction

Preregistration5f3cc06bb12c438c8c807af7d2a4273447122a85 precedes all real fits.
Implementation67fcdbc6af761c970292a06b11aa0deb6ff6567f passes source CI37652057020.
Local18 new tests and the full1,178-test suite pass (159 existing warnings), with
critical Ruff. Tests cover emit-before-update, last-finite missing/censored time,
future/prefix causality, episode reset, shared frozen transforms, original weights,
analytic sequence gradients, fixed firsts and unsupported chronology. All9 full/
fitted-fold heads converge within the unchanged normalized gradient<=1e-6 rule.

Before official execution, two-artifact packaging is registered because the
unchanged eight-table output exceeds the transfer archive limit. This changes
serialization/packaging only, not models, tolerances or evaluation. Single cached
official run37652470621 succeeds from6c2bb5518187c41d7a59a3e1d9d3085fd9ec50dd;
its source full CI37652475906 also succeeds. No failed official R327 attempt or
scientific retry. Both artifacts are source/digest authenticated:

| Artifact | ID | ZIP SHA256 |
|---|---:|---|
| Primary result/evaluation/ledgers | 11497031370 | 206485c8c0326b0ef6010de366fa1b4ebdc4164f89556c7a9a1dcf66613929b3 |
| Training/feature histories | 11497126349 | c41041d2ebabfb0b97fb4825d7de4639da7cb5ba92516ff42837d18869e4c60c |

Duplicate source/environment files are byte-identical before merge. Main JSON
5,278 and fit-audit3,663 numbers have maximum error0; main JSON bytes are identical.
All eight tables, including35,142 feature-history rows, have numeric error0,
exact identities/labels/missingness, tie-aware score ordering and rate ordering.
Tolerance remains1e-9 with no relaxed tie/missingness rule. Results, both local/
official JSON copies, fit audit, reproduction and provenance are saved under
research/results/request327*.json. Cached parquet tables remain the official
artifacts; the executed workflow is recoverable at the source commit and retired
after verified replay. Draft PR100 remains unmerged.

## Next experiment

R328 will test a **causal availability-gated trajectory residual on frozen G**.
This follows the observed availability contrast as a new hypothesis, not a claim
that its cause is proven. Keep absolute-quality G fixed; compare seven EMA,
seven first-anchor and seven current-feature corrections plus a mask-only
calibration control on the exact same gate. Unavailable-history predictions copy
G exactly, while every original row/pair and loss weight remains included.

Training offsets must use strict-past canonical G, never an in-sample full-fit
R326/R327 meta prediction. Training-only audit finds183 active episodes/41
positive episodes across3 dates, with6,546 complete targets and original mass
130.131680 within mass256. Details, fixed optimizer/40 gates and honest unsupported
May6 handling are in request328-proposal.md and request328-training-support.json.
No real R328 fit/evaluation has run. No policy, raw acquisition, sealed-date access,
main merge, live trading or promotion is authorized by this development result.
