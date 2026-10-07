# R322 official findings: causal trajectory bridge does not improve admission

Primary B fails15/20 preregistered checks. Its SAME-day BETWEEN-episode rank is
.624955 versus matched baseline D .624728, with both cluster intervals including
zero. The learned relative timing representation has not established incremental
absolute admission value. Do not promote B, pick another arm as a fallback, or
retune windows/features/regularization on these outcomes.

Registrationae9af04682ea5a897291a9e99bafaeb4a9ca8cf7 preceded real meta fits
and evaluation. Parent R32127f43c7dcced1cd8b09a25b66e6ad00ff7ac81a3, Draft
PR94. Official R322 source3d8e996e7ea327451d8d575776294271d430efbe,
run37596148558, artifact11470773466 succeeded. ZIP digest
1ef656d482875b75fdf6cd96d6dd9f6f21c397003587e7df626114086dad037c.
Draft PR95 targets R321; no main configuration, costs/stops, market acquisition,
sealed June HOLD/July-August, promotion or trades changed.

## Scope and model contract

Meta-training7327 complete states from256 episodes/51 positive episodes on
May6/7/8, with original independent weight mass256. Original7806 observations
remain available to causal history, including479 censored observations. Saved
W/A heads and each normalizer use strictly preceding dates. Stage-one pair
support is8/25/44 mixed episodes; normalization uses80/175/259 complete preceding
episodes. Evaluation uses the frozen full May5-8 heads and336-episode training
normalization. This training-support shift remains a limitation, not a confirmed
explanation for failure. All raw held scores replay exactly in the local run.

The three learned features are current normalized score, change from first
observed score and change from latest observed score<=t-30s. History is formed
before censor filtering; missing lag stays missing. Each first/lag input exists
at the current decision clock. No future mean, forward fill or own-day fitted
timing feature. Four fixed classifiers use identical target support/weights and
learner parameters, with each preprocessing fit only on meta-training states.

New probability columns use the R322_ prefix because inherited R319 C/C0 columns
already exist. The first local execution caught this implementation collision
before producing evaluation results. Namespacing preserves every old score and
does not change the registered arm semantics, feature packages or parameters.

## Official evaluation

Keep828 identities in the ledger,19530 observations/18793 complete states and655
complete episodes;78 mixed episodes inform conditional timing. Primary between-
episode rank excludes comparisons within the same original episode.

| Arm | Between-episode same-day AUROC | Pooled AP | Brier | Within-episode AUROC |
|---|---:|---:|---:|---:|
| B Q/history + learned W trajectory, primary | .624955 | .082940 | .052212 | .521717 |
| C Q/history + elapsed A trajectory | .622570 | .078622 | .052257 | .524266 |
| D matched Q/history baseline | .624728 | .079401 | .052161 | .528191 |
| E matched raw M/history/lag baseline | .610860 | .072249 | .052608 | .571490 |
| B0 own first probability on same later labels | .635438 | .093826 | .054702 | .500000 |
| Each new head's own training-prior baseline | — | — | .051078 | — |

All four new probability heads have worse Brier than their prior. B's small AP
point improvement does not override failed rank/calibration gates. B-D rank
improves on4/8 days, not a majority. First-clock B rank .641591 versus D .637908
is descriptive only, not an alternative selected endpoint or a new gate.

| Primary rank difference | Point | Day95% interval | Ticker-day95% interval |
|---|---:|---|---|
| B-D | +.000227 | [-.004874,.006770] | [-.005018,.005344] |
| B-C | +.002386 | [-.004315,.007192] | [-.002930,.007352] |
| B-E | +.014095 | [-.010730,.052111] | [-.008215,.037599] |
| B-B0 | -.010482 | [-.029049,.018457] | [-.027572,.008219] |

B-D Brier difference+.0000508 (worse): day interval[-.0000598,.0001642],
ticker-day[-.0000892,.0001940]. B-C Brier difference-.0000445: day
[-.0001427,.0000649], ticker-day[-.0001793,.0000991]. All1000 draws per scheme
are valid. Fixed-fit intervals omit uncertainty from both learned stages.

Passing5/20 checks: training>=200 episodes,>=30 positive training episodes,>=4
positive evaluation dates, B AP>C/D/E, and B timing>.5. Failed15/20: combined
primary-rank superiority, combined Brier superiority, majority dates, all eight
positive rank CI checks and all four negative Brier CI checks. No criterion is
relaxed. Saved external references have different fitting support: Q rank.628529,
G .618618, X .643225, X0 .647708; they are not new fallback models.

## Verification and implication

Full final local suite1093 tests passes; critical Ruff passes. Official cached
replay and source-commit full CI both succeed. All1432 JSON numeric fields reproduce
within2.6645352591003757e-15, far below registered1e-9. JSON bytes and the output
meta parquet digest differ; each digest is independently checked against its OWN
checkpoint bytes. The verifier compares the remaining JSON contract and every
parquet value within the original tolerance. Four checkpoints retain identical
identities, labels, clocks, missingness and every tie-aware score ordering:
meta7806x108, all states19530x122, first observations679x122, ledger828x13.
Maximum frame error2.6645352591003757e-15. Exact official/local JSON plus audit are
committed; full parquet artifacts remain in the official run. One-shot workflow
is retired after verification.

R321's price/activity timing beyond its elapsed control remains supported on
known May development, while R322 fails to transfer that signal into absolute
admission. The W-current meta coefficient is+.085479, from-first-.055527, lag
change-.001820 in processed units; these are descriptive fitted coefficients,
not causal importance or proof that the lag signal is absent. A loss/task mismatch
and small chronological training support are plausible next questions. The next
separate study should test a fixed joint absolute/relative objective with matched
feature/support/weight controls; preserve both admission and timing checks.
