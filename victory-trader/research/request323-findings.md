# R323 official findings: joint loss improves timing but worsens admission

The primary fixed1:1 joint linear learner J passes12/21 checks and fails9/21.
Whole-episode timing improves .480807→.643724 versus same-feature state-only C,
and local30s pair rank .463408→.584579, with both cluster intervals positive.
Between-episode admission rank worsens .590160→.548755 and Brier worsens
.053191→.055982. Do not adopt J or select a secondary arm/mixture weight as fallback.

This tested shared linear score creates a tradeoff. It does not establish that
every joint/nonlinear/multitask architecture must fail. Pair training is useful
for conditional timing but does not automatically make an absolute probability
score. The prior best frozen W remains stronger at whole timing .671630 and local
.599223; J loses-.027906 and-.014644 respectively even against that reference.

## Registered and reproduced scope

Registration4fe63d58219cbe812875a3f87ec26d9ad4be4c8b preceded all new real fits
and R323 evaluation. Parent R322d3c5b82d7d096e14171b8f6484c90a68a3a43b97,
Draft PR95. Official source68a15980cb9e99cafb3870730c3407cbf710caa5,
run37599496559, artifact11471643549 succeeds. ZIP SHA256
7f55b43b9d84cb6d93036d1ccf493d75ec955db22835b71d1e16d7209993ab82.
Draft PR96 targets R322. Standing routine research authorization applies; main
configuration, labels/costs/stops, market acquisition, sealed June HOLD/July-August,
promotion and trades remain unchanged.

Same9325 training observations/8792 complete states/336 complete episodes,
62 positive episodes,50 mixed episodes/all47825 inherited pairs. Joint objective
has absolute component mass168 and rescaled relative component mass168 at fixed
total336 and C=.1; pairs retain independent mixed support50. J-C changes that
objective mixture, not merely a feature, and not a causal market treatment.
Both rich heads copy exact R319 T38 transforms, context heads exact G12 transforms.
All transforms replay training scope; all fixed optimizers converge with normalized
gradient infinity norm<=2.71e-8, below declared1e-6. State-only C probabilities
replay saved T within3.731923103877932e-11 and D replays G within2.7755575615628914e-16.

All828 evaluation identities remain,19530 observations/18793 complete states/655
complete episodes. Conditional whole timing uses78 mixed episodes; the inherited
local diagnostic uses70 episodes/all8623 pairs across8 dates. These are not47825
or8623 independent examples. Censored observations, original scores, missingness,
clocks, all pair manifests and ledger identities stay unchanged.

## Official metrics

| Arm | Between-episode same-day AUROC | Within-episode AUROC | Local30s pair rank | Brier |
|---|---:|---:|---:|---:|
| J rich joint, primary | .548755 | .643724 | .584579 | .055982 |
| C rich state-only matched | .590160 | .480807 | .463408 | .053191 |
| H context/history joint | .551842 | .563719 | .510683 | .054134 |
| D context/history state-only | .618618 | .532915 | .484978 | .051957 |
| J0 own first score on same later labels | .576466 | .500000 | .500000 | .058947 |
| W frozen R320 pair-only reference | .467818 | .671630 | .599223 | not a probability |
| All new heads' own prior | — | — | — | .050811 |

J pooled AP .074850 versus C .065378, H .065008 and D .076236. This AP gain over
C does not override failed primary/calibration checks. All joint/state sigmoid
heads lose to their own prior on Brier. J's absolute Brier worsens even though the
intercept is fitted on absolute targets; a sigmoid link does not prove calibration.

| J-C difference | Point | Day95% interval | Ticker-day95% interval |
|---|---:|---|---|
| Between-episode admission AUROC | -.041405 | [-.090523,.020579] | [-.108677,.023266] |
| Whole-episode timing AUROC | +.162917 | [.108741,.225980] | [.085829,.243348] |
| Local pair rank | +.121171 | [.045306,.190819] | [.042380,.203641] |
| Brier (positive=worse) | +.002791 | [.000796,.004914] | [.000502,.004779] |

Compared with context state-only D, J admission delta-.069864 has day interval
[-.131393,-.010373] and ticker-day[-.143693,-.002849]. This loss is supported by
both schemes. J-H admission delta-.003088 has intervals including zero. J-C
admission improves on2/8 dates, failing the majority criterion. Both schemes have
1000 valid draws for every metric; intervals condition on fixed models and omit
fit uncertainty.

Twelve passing gates: all six support checks; J timing>C/H/.5; J local>C/H/.5;
both-scheme positive whole-timing J-C intervals and both-scheme positive local
J-C intervals. Nine failures: J admission>C/H/D, J Brier<C/H/D/prior, majority dates,
both-scheme positive admission J-C intervals, both-scheme positive admission J-H
intervals, and both-scheme negative Brier J-C intervals. No gate is relaxed.

## Validation and next mechanism

Full local suite1103 tests and critical Ruff pass. Tests verify analytic gradients,
sklearn state-only boundary, independent state/pair mass, orientation equivalence,
intercept cancellation, causal one-state inference, scope and censoring. Official
replay and source-commit full CI succeed. All1819 JSON numeric fields reproduce
exactly; JSON bytes are identical. All seven parquet checkpoints are exactly equal,
including labels/clocks/missingness and every tie-aware score ordering. Full
artifact ZIP digest verifies its bytes; exact official/local JSON and reproduction
audit are committed. One-shot workflow is retired after verification.

Next model should distinguish initial opportunity assessment from conditional
state evolution instead of forcing both into shared coefficients. An exploratory
TRAINING-ONLY audit, performed after R323 to assess feasibility and before any
next-model fit, finds8445 adjacent complete transitions/286 episodes:0→1 has88
transitions/38 episodes;1→0 has99/48, across all4 training dates. The full audit is
request324-training-support.json. These are hindsight label-state transitions,
not observed trades/profits or market regimes. No R324 model or evaluation was run.
The next proposal tests a recursive latent-opportunity model with prediction-only
state at inference and matched drift/context/current-state controls; see
request324-proposal.md. Reused May evidence cannot justify promotion or sealed data.
