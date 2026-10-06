# R315: matched population and canonical first-observation training

Preregistered before new canonical training labels, fits or market acquisition.
Continue from R314 without changing features, risk rules or evaluation cohort.
Previously reused May is development; June HOLD and July–August stay sealed.

## Population and acquisition

Read only R311 POPULATION_COLUMNS from R233 run36251239308 and the independent
R163 reference run36001945920, filtering dates at parquet read. Cross-check
identities and context fields. Use May5,6,7,8 (CROSSFIT_DAYS) for training and
the original R311 eight May11–20 dates for replay only. Upstream fit boundaries
remain the previously audited Jan16 attention / Jan20–Feb9 second reranker.
Do not reuse setup/economic admission decisions, targets or outcome columns.

B training cohort: whole upstream HOT population on May5–8 sampled by unchanged
SHA256('R311|day|uppercase_ticker|HOT_ms').digest()[0]<64. No quota, replacement,
price/winner filter or outcome stopping. S comparator: the original97 R310
selected identities with source HOT context, at the same canonical clock.
Persist union membership and cohorts before acquiring or labeling. Fetch only
declared union May5–8 ticker-days, reusing validated R306 raw history where
available and the existing cache elsewhere. Missing/rejected/empty histories
remain explicit; no borrowing observations from another date or later clock.
No evaluation-market requests or new dates. Log cache/network/retry statistics.

Use FIRST completed regular one-second aggregate starting at/after HOT and
before HOT+300s, with decision before HOT+1-hour/session cap, exactly R311.
Label with unchanged next-open entry, BASE costs, -2.5% absorbing stop and
pending terminal fill. Preserve unresolved labels and do not replace them with
zero or move to a later observation. Only complete available training labels
enter fits; report their cohort coverage and resulting selection.

## Fixed model comparisons

R: replay original M28 fit on4,526 R310 states, seed20264605, saved R311
probabilities within1e-9. S: M28 fit on canonical first observations of selected
97. B: M28 fit on sampled full-HOT canonical first observations. Q: original
Q9 controls fit on the SAME B rows. All use unchanged fit_linear transforms,
C=.1, lbfgs, tolerance1e-8, max_iter2000, independent episode/day weights and
single-class prior fallback. S/B/Q seed20265005. No feature, threshold, model,
seed or calibration search; no choosing a successful head as a fallback.

Evaluate R/S/B/Q on the unchanged828 R311 identities at original first clocks;
primary final-label metrics remain the original650 complete cases. Score all
679 observed cases regardless of label availability. Primary B−S contrast
tests the selected-versus-broader training population at matched clock/input
contracts; B−R is total training-contract intervention, B−Q is momentum
increment. They are fixed development comparisons, not causal treatment effects.
Canonical S may differ from the original selected first saved clock and label;
report those transitions rather than calling S−R a pure clock effect.

Chronological training diagnostic: May6 fits May5, May7 fits May5–6, May8 fits
May5–7, restricted to sampled B labels available before held day. B/Q preprocessing
uses these earlier days only; May5 stays unscored without earlier broad history.
No current/later-day fit, earlier evaluation-label update or leave-one-day-out
future-day training. Abort if a required frozen/chronological fit has zero rows.

## Metrics, gates and checkpoint

Day/episode-weighted same-day AUROC is primary. Report pooled AUROC/AP/Brier,
training-prior Brier, per-day metrics, positives and coverage. Fixed-score paired
day and ticker-day bootstrap1,000 draws, seeds20265050/20265051; contrasts
B−S, B−R, B−Q. Count invalid rank draws. No fitting-uncertainty claim.

Fixed checks: evaluation coverage>=90%, >=20 positives, >=4 positive days;
B same-day AUROC beats S/R/Q; B AP beats S/R/Q; B Brier beats S/R/Q and B prior;
B−S improves AUROC on majority evaluable dates; ticker-day same-day lower CI
positive separately for B−S and B−Q. No promotion even if all pass, no threshold
relaxation or final/sealed-date access. Training coverage reported independently.

Save source/cohort hashes, unchanged evaluation scores, all training union states,
raw history and pre-acquisition membership, fit/fold audits, missing reasons and
gate checks. Local tests use synthetic clients; official acquisition uses existing
configured credentials. Download official checkpoints, rerun entirely offline
and compare numbers (1e-9), identities, labels, ranks and flags before reporting
completion. No duplicate experiment required for local reproduction.
