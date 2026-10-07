# R322: preceding-day trajectory representation for absolute admission

Register remotely BEFORE real-data meta fits/evaluation. Parent official R321
closeout27f43c7dcced1cd8b09a25b66e6ad00ff7ac81a3/Draft PR94. R321 confirms
nearby price/activity discrimination beyond its elapsed control, while absolute
between-episode admission remains unresolved. No static interaction search.

## Frozen scope and causal representation

Seven files from official R319/R320/R321 are pinned in request322-inputs.json.
No market acquisition, new labels/clocks, sealed June HOLD/July-August, costs/stops,
promotion or trades. Same reused-May development, not independent confirmation.
Keep all828 evaluation identities/19530 observations/18793 complete states.

Meta-training uses ONLY official chronological R321 states for May6/7/8:
7806 observations,7327 complete states,256 complete episodes/51 positive episodes.
The frozen W and age-only A pair heads for each date were fitted on preceding
May dates only (8/25/44 mixed episodes). No own-day fitted score or own-day learned
normalization enters a meta-training observation. May5 is excluded from meta-
training because it has no preceding timing fit, not because of its outcomes.
No cross-validation choice, extra seed, regularization or feature search.

For each frozen head/fold, reconstruct its training decision scores using saved
transform/coefficients on ALL COMPLETE preceding training states. Normalize raw
scores with their original equal-day/equal-episode/state weighted mean and standard
deviation, using only that head's preceding fit days; if std<=1e-12 use1. Verify
saved held raw score replay within1e-9. The full May5-8 head/normalization is used
for May11-20 evaluation; never fit normalization on evaluation states.

Create exactly three features for EACH W/A representation: normalized CURRENT
score; CURRENT minus own FIRST-observation normalized score; CURRENT minus latest
observed normalized score at clock<=t-30s. Reuse the30s lag without search. Build
features on ALL original observations BEFORE censor filtering; missing lag stays
missing, with no forward fill. First/current/lag scores at their own clock use a
frozen past-trained head. No future episode mean, future score or future label.
These features encode learned trajectory, not a new static feature interaction.

## Four fixed absolute probability heads

| Arm | Features | Role |
|---|---|---|
| B | R319 G (Q9+history) + three W trajectory features | primary improvement |
| C | identical G + three A elapsed trajectory features | matched elapsed control |
| D | G only | matched admission baseline |
| E | R319 T (M28+history+seven raw lag differences) | matched raw-feature baseline |

Each fits the SAME7327 complete meta states/256 episodes and identical independent
weights (sum256), training-only1/99% clipping, weighted median/mean/scale and
missing flags. LogisticRegression(C=.1,l1_ratio=0.,solver=lbfgs,fit_intercept=True,
max_iter=2000,tol=1e-8,random_state=20265605). All inherited net>=5 labels unchanged.
Copy own training prior; broadcast own FIRST score B0/C0/D0/E0 onto the SAME later
labels as temporal controls. Saved R319 Q/G/X/X0 ranks remain external descriptive
references, with their different support/objective explicit. Do not select another
arm when primary B fails, or promote a diagnostic-only improvement.

## Fixed metrics, intervals and gates

PRIMARY SAME-day BETWEEN-episode AUROC on every complete evaluation observation,
excluding same-original-episode pairs. Original equal day/episode/state weights.
Report pooled AUROC/AP, Brier versus own prior, within-episode timing AUROC and
first-observation metrics. No threshold, top-k entry policy or realized returns.
Bootstrap1000 whole-day/ticker-day multiplicities on ORIGINAL weights, seeds
20265650/20265651. Fixed-fit intervals omit first/second-stage fitting uncertainty.
Compare B-D, B-C, B-E, B-B0, C-D and E-D on all five metrics. Retain all rows,
censoring and828-identity ledger; coverage must not be increased by dropping cases.

Twenty gates: >=200 complete training episodes; >=30 positive training episodes;
>=4 positive evaluation days; B primary rank > C/D/E/B0; B AP > C/D/E;
B Brier < C/D/E and own prior; B timing >.5; B-D rank improves on a majority of
evaluation dates; positive DAY and TICKER-DAY lower rank CI for B-D/B-C/B-E/B-B0
(eight checks); negative upper Brier CI for B-D/B-C in both schemes (four checks).
All are kept, no gate relaxation. Passing supports further development only,
never promotion on repeatedly reused May. Isolate incremental W representation
against age C; raw E prevents attributing ordinary raw feature capacity to the
learned bridge. B versus B0 compares updated and fixed-first scores on same labels.

Persist all enriched meta/evaluation states, first clocks, fits/normalizers and
audits. Synthetic tests cover preceding-day scope, replay, causal first/lag clocks,
no forward fill, missing/censored history retention, model allowlists and independent
weight mass. Full tests and critical Ruff; one official cached reproduction after
local validation, compare JSON/parquet/missingness/tie-aware score ordering within
1e-9 and retire one-shot workflow. Keep main/research-request.json unchanged.
