# R328: quality is preserved more closely; EMA innovation remains weaker than current evidence

R328 F passes **13/40** preregistered development checks. EMA innovation improves
local timing over frozen G and mask-only M, but is weaker than current-coordinate
S on whole/local timing and anchor innovation N on whole timing under both
bootstrap schemes. Admission and probability-error improvement remain unproved.
All27 failures remain; no head is selected or promoted. Official reproduction
passes after an independently reconstructed local serialization repair.

## Fixed architecture and support

Keep strictly out-of-time G as a fixed logit offset; fit7 penalized slopes for
EMA innovation F, first-anchor innovation N and current coordinates S, and1
penalized gate scalar M. No intercept/missing coefficients for F/N/S. All four
share the causal all-seven-history gate. Gate0 and all original firsts copy G bits
exactly; every original row/pair remains. G/B0 are unchanged current/own-first G.
S uses current coordinates already included in G, not features stripped of history.

Training is7,806 original May6–8 observations/7,327 complete targets/mass256.
Gate1 has6,546 complete targets/183 episodes,41 positive episodes and original
mass130.1316797413857. Gate0 has781 complete targets/mass125.8683202586143, with
constant loss and zero gradient. Preprocessing remains7,066 original transition
currents/mass218, including inactive rows. All heads use C=.1/one zero start and
the fixed optimizer; full/fitted folds require normalized gradient infinity<=1e-6.

Evaluation keeps19,530 observations/18,793 complete targets,828 identities/655
complete episodes,78 mixed/70 local episodes and all8,623 pairs across8 May dates.
Labels, censoring, clocks, costs/stops and sealed dates remain unchanged. These
hindsight reachable-net targets and ordering metrics are not realized profit.

## Development results

| Arm | Admission | Whole timing | Local timing | Brier |
|---|---:|---:|---:|---:|
| F: EMA innovation | 0.617602131 | 0.535146170 | 0.549400948 | 0.051975664 |
| N: first-anchor innovation | 0.622897404 | 0.584612241 | 0.548020772 | 0.051989921 |
| S: current coordinates | 0.617945735 | 0.604439329 | 0.611142359 | 0.052091506 |
| M: gate scalar | 0.624292831 | 0.540548014 | 0.474031295 | 0.052618616 |
| B: current frozen G | 0.618618247 | 0.532915297 | 0.484978269 | 0.051957113 |
| B0: own-first G | 0.639103990 | 0.500000000 | 0.500000000 | 0.054755010 |

| Contrast / metric | Point difference | Day95% CI | Ticker-day95% CI |
|---|---:|---|---|
| F−B admission | -0.001016116 | [-0.004476252, 0.002762167] | [-0.003653425, 0.001496653] |
| F−B whole | +0.002230874 | [-0.013163668, 0.015739875] | [-0.013940784, 0.017328230] |
| F−B local | +0.064422679 | [0.012963685, 0.106184652] | [0.017620774, 0.119850067] |
| F−B Brier | +0.000018550 | [-0.000049227, 0.000097212] | [-0.000054504, 0.000099433] |
| F−N whole | -0.049466071 | [-0.074602999, -0.029874053] | [-0.078783983, -0.023702316] |
| F−N local | +0.001380176 | [-0.044766504, 0.056321495] | [-0.054361137, 0.062111172] |
| F−S whole | -0.069293158 | [-0.110496912, -0.036678285] | [-0.102008706, -0.039397426] |
| F−S local | -0.061741411 | [-0.105536649, -0.020557331] | [-0.115238827, -0.005023040] |
| F−M local | +0.075369653 | [0.003535954, 0.139282678] | [0.017518513, 0.138192873] |
| F−M admission | -0.006690700 | [-0.011839371, -0.001112501] | [-0.013322202, -0.000202979] |
| F−M Brier | -0.000642953 | [-0.000899822, -0.000412323] | [-0.000991294, -0.000268920] |
| S−B whole | +0.071524032 | [0.034998826, 0.114804245] | [0.034937225, 0.109983568] |
| S−B local | +0.126164091 | [0.048319963, 0.195220407] | [0.062216319, 0.192083629] |
| N−B whole | +0.051696944 | [0.028757839, 0.081461959] | [0.022413764, 0.082598354] |
| N−B local | +0.063042503 | [0.017456712, 0.104800824] | [0.007088467, 0.118820093] |
| M−B Brier | +0.000661503 | [0.000496008, 0.000866512] | [0.000319674, 0.000989703] |

Nine support/integrity checks pass, plus both-scheme F−B local and F−M local.
All four aggregate checks fail, as does the majority-date admission check: F
beats G admission on only2/8 dates. F−B whole/admission/error intervals contain0.
F's Brier excess versus G is only .000018550, much smaller descriptively than
R327's .000652816, but this changes several architecture/objective choices and
does not identify a gate-only cause. Brier measures probability error, not its
calibration component in isolation.

Current-coordinate S beats G whole/local timing under both interval schemes;
N also does so. S/N admission and Brier improvements over G are not established.
F−S whole/local and F−N whole intervals are strictly negative. Thus this specific
EMA deviation representation loses information useful to the matched controls.
This does not refute all observed trajectories or authorize selecting S/N after
the primary hypothesis fails. No rate, τ, gate, feature, penalty or optimizer search.

Fixed-fit1,000-draw whole-day/ticker-day intervals retain original independent
weights and omit model fitting/gate-design uncertainty. This is reused May
development, not fresh validation.

## Gate behavior and correction strength

Gate1 contains17,646 observations/16,980 complete targets from460 complete-target
episodes, original mass332.245570. Gate0 contains1,884 observations/1,813 complete
targets from653 episodes, mass322.754430; every F/N/S/M gate0 prediction is EXACT
G. Episode counts overlap; row/weight partitions add. Slice metrics rescope their
own weights and cannot be added to obtain the full result.

Gate1 local F/N/S/M/G ranks are .563908/.582037/.652617/.464651/.464651 over57
local episodes. Gate0 all four heads/G have local .479650 over18 episodes. These
are descriptive slices, not selected validation or identified missingness causality.
M is a constant positive gate1 logit correction .167455328; it preserves ranks
inside the gate1 slice but shifts active-versus-inactive comparisons in the full
population. M's probability error is worse than G under both schemes.

F's original-weight active logit correction has median .000278841 and10/90th
percentiles [−.061246089,.063769147], compared with S [−.257466497,.215365478]
and N [−.260787639,.257834632]. This describes small fitted EMA corrections, not
proof that their magnitude alone causes the failure or a reason to force it larger.

## Chronological check

May7 trains only May6; May8 only May6–7. May6 has no preceding meta fit: all4,602
observations remain,284 gate0 predictions are exactly G and4,318 active F/N/S/M
predictions are unsupported. Full chronology keeps7,806 rows/3,488 common finite
rows; the extra finite inactive rows are a defined architecture branch, not an
invented fitted head. Fitted May7/8-only has25 mixed/23 local episodes.

| Arm | Admission | Whole timing | Local timing | Brier |
|---|---:|---:|---:|---:|
| F | 0.748517706 | 0.560202276 | 0.582284074 | 0.086020044 |
| N | 0.750266888 | 0.559572506 | 0.596429656 | 0.085577915 |
| S | 0.753639493 | 0.573318593 | 0.610528028 | 0.085988223 |
| M | 0.753014635 | 0.561644013 | 0.600956199 | 0.086334518 |
| B | 0.752820012 | 0.556293723 | 0.595865701 | 0.085850686 |
| B0 | 0.791997466 | 0.500000000 | 0.500000000 | 0.082181407 |

F's full-development local gain over G does not transport to this small fitted
chronological slice: F local .582284 trails G .595866. S local .610528 remains
higher by point estimate, while F admission and Brier trail G. Report these
limitations alongside the full comparison, not as a new validation period.

## Validation, execution and serialization recovery

Preregistration38e3c0f529230f035c5101a423b86c9ebb3b978f precedes all real fitting.
Implementation5af41275ba05ac12b0f3434e6f5b32d70419aa16 passes CI37657051067 and
37657058800. All18 new tests and the1,196-test suite pass;159 existing warnings
plus1 expected unsupported-NaN logaddexp warning. Critical Ruff passes. All12
full/supported-fold heads converge; maximum normalized gradient9.492001615e-9.
Tests cover offset gradients/constants/censored NaN, missing-history gate,
exact G copies, future/prefix/label invariance, preserved weights, unsupported
folds and a failed solver audit without retry.

One official cached run37657314310 succeeds from source
aa031411f1ab790c2c10d5f0a92dc9aa9225aead, whose full CI37657328873 also succeeds.
Both artifacts/source are authenticated; duplicated source/environment files
are byte-identical before merge. Two-artifact packaging was registered BEFORE
fitting and all nine tables remain required.

| Artifact | ID | ZIP SHA256 |
|---|---:|---|
| Main result/evaluation/design/ledgers | 11499476294 | a5824927b52defc0521ba78edf01e1d69bf2b55df928798788d9dca12f9da222 |
| Training/feature histories | 11499116858 | ce72b09fb11e99b0bbb048b642d99ebb521fac5e0e7b8d4069027d56e30e0467 |

Initial verification stops on a truncated LOCAL design-matrix parquet (4,393,450
bytes, no footer). It is an exact prefix of the official5,470,678-byte file; the
other eight local tables are already byte-identical to official, and main/fit
JSON numbers have error0. Root cause is not established. Preserve the original
truncated file and request328-initial-verification-failure.json.

Recover ONLY serialization of the registered design table from the independently
saved LOCAL training/evaluation/chronological state tables, original columns and
original weights. No official values are copied into the reference, no new fit,
scientific contract change, fit selection, optimizer retry or tolerance/rank
relaxation. The recovered file is also byte-identical to official. Second complete
verification passes: main JSON2,050 and fit audit705 numeric fields have error0;
all nine tables (both audit tables35,142 rows each) have error0, exact identity/
label/missingness and score/correction/logit tie-aware ranks. Main JSON bytes and
all nine parquet bytes match. This serialization recovery is explicitly recorded,
not hidden as an uninterrupted first-pass validation.

Full results/local/official copies, fit audit, failure/reproduction/provenance are
under research/results/request328*.json. The executed one-shot workflow is
recoverable at the source commit and retired after verified replay. Draft PR101
remains unmerged. No raw acquisition, policy change, sealed-date access or trading.

## Next experiment

R329 tests accumulated **current-evidence level**, using the verified current S
numeric transform and the same G offset/gate/weights. Compare a fixed30s causal
EMA state against a first-eligible anchor, while freezing R328 S/M as current/
mask controls. First eligible evidence initializes both states; inactive clocks
carry state/time while keeping G-only predictions. This asks whether memory of
observed evidence helps beyond its instantaneous level, without static interaction
search or choosing a winning fallback arm.

Training-only support has162 carried-history complete-target episodes/38 positive
episodes over3 dates,6,364 complete targets/mass113.746935 within original mass256.
Details and fixed40 checks are in request329-proposal.md and the training audit.
No real R329 fit/evaluation has run. Add complete-footer/atomic serialization
checks before its large outputs to guard against the observed local truncation.
