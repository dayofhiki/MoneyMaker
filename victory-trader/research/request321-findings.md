# R321 official findings: nearby price/activity timing survives elapsed control

All nine preregistered diagnostic gates pass. This establishes a reused-May
development signal beyond the specified elapsed-only control, not a promoted
model or executable trading policy. It does not establish incremental seven-lag
features or superiority of pair loss on nearby pairs.

Registrationf4e380ac75c117363fddbfe1d0f1b5c071d56035 preceded all new fits and
nearby evaluation. Official source1d463371a9e28a854eaf6eeabe0022e57651f2be,
run37594980002, artifact11470147064 succeeded. ZIP SHA256
101749d20d59d0523f94a6c0c66e501c4b6a894e0c134aac66ee2d3da657aa0e.
Draft PR94 targets R320. Standing authorization remains in effect; no sealed
dates, acquisition, costs/stops, main configuration, promotion or trades changed.

Same50 mixed training episodes/47825 inherited pairs/total fitting weight50.
One new A head uses only three history support inputs plus missing flags. Its
transform copies the exact corresponding training parameters from R320 W.
All R320 heads, inputs, labels and censored states stay fixed. The official
19530-state checkpoint retains all observations, and its ledger all828 identities.

The priority30s diagnostic has8623 complete positive-negative pairs from70
episodes across all8 evaluation days, weighted equally by eligible day/episode,
then by pair within episode. These pairs are not8623 independent samples.

| Frozen/control arm | Whole-episode timing AUROC (78 episodes) | Local pair rank (70 episodes) |
|---|---:|---:|
| A elapsed-only pair head | .525409 | .499506 |
| U Q9+history pair head | .575280 | .526281 |
| V M28+history secondary ablation | .696270 | .581097 |
| W M28+history+seven lag differences, primary | .671630 | .599223 |
| Z matched endpoint state classifier | .627509 | .607417 |
| T saved R319 rich state classifier | .480807 | .463408 |
| W0/A0 constant-first controls | .500000 | .500000 |

Local pair rank and whole-episode AUROC use different pair membership and episode
weights; do not subtract these columns or call the local result unconditional
AUROC. Conditioning on future labels is for evaluation only, not online eligibility.

| Local difference | Point difference | Day95% interval | Ticker-day95% interval |
|---|---:|---|---|
| W-A | +.099717 | [.055475,.146158] | [.032950,.165247] |
| W-U | +.072942 | [.034677,.115623] | [.006228,.142041] |
| W-V | +.018126 | [-.015245,.051355] | [-.017169,.056046] |
| W-Z | -.008194 | [-.038739,.026888] | [-.044605,.036632] |
| W-T | +.135815 | [.079363,.190875] | [.043061,.224352] |
| W-W0 | +.099223 | [.056776,.143372] | [.017214,.176567] |

All1000 draws in each scheme are valid. W-A improves on7/8 dates; only May11
decreases. Gates pass: training>=30, local evaluation>=30, dates>=4, W>A/U/.5,
W-A majority dates, and both-scheme positive intervals for W-A and W-U. Large
coefficients on time do not by themselves explain away the timing result. The
age-only head has near-chance local ordering, while price/activity timing survives.
Other age/quality confounds remain possible. W's seven extra lag features have
uncertain increment over V, and pair loss does not beat Z in this local diagnostic.

Full local suite1085 tests passes; critical Ruff passes. Independent official
run reproduces all465 JSON numeric fields with maximum error0 and identical JSON
bytes. All four parquet checkpoints match exactly, including missingness, labels,
clocks, pair membership and every tie-aware score ordering. Audits and exact
official JSON are committed; the official artifact retains full parquet data.
The one-shot workflow is retired after verification.

Next model question: whether a frozen relative timing representation improves
absolute remaining-opportunity admission on ALL complete episodes, rather than
only mixed episodes. A separately preregistered two-stage bridge must use
preceding-day timing scores for meta-training, training-only score normalization,
causal current/first/lag changes computed before censor filtering, matched
elapsed-trajectory and raw-feature controls, and full out-of-date evaluation.
No own-day fitted timing score may enter meta-training. This is the current
bottleneck; another static-interaction search is not indicated.
