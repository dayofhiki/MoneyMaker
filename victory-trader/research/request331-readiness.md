# R331 prepared continuation — experiment not executed

This is the preserved PRE-EXECUTION snapshot. R331 is now executed and exactly
reproduced:9/10 checks pass, while34.8352% execution resolution fails the90%
requirement. Current scientific status and source/artifact audit are in
request331-findings.md and request331-provenance.json; R332 is a designed,
unexecuted cached source/execution observation proposal.

Historical preparation snapshot: the user explicitly approved public R331
design/code/hash publication and official cached execution on2026-10-08, then
renewed same-scope standing approval. The two earlier review rejections below
remain in the audit; publication/execution will be recorded in final findings.

Latest completed Work experiment is R330, closed on the remote branch at
6c72824d36a5f86d4f282b0cb5f201d1d136eac7. Its 8/18 passed gates do not establish
a profitable trader: F/E make zero entries; the 104 common-target training
episodes fall below 150. Removing WAIT availability alone still leaves only
114 immediate-target episodes. Explicit memory remains low priority.

The next proposal has been converted into a concrete local preregistration and
implementation on `research/action-support-census-331`. Local preregistration
commit is fa894b68e9eceb3e09ae6aebca1994042351c1db. **This commit is not remotely
registered. No R331 market-data clocks, labels, fits or evaluations were run.**

## Fixed prepared experiment

- Keep all 407 original broad May5–8 training identities and all 828 May11–20
  evaluation identities. Primary support scope is the 305 May6–8 identities.
- Retain original first-five-minute clocks; add only the first completed print
  in each fixed 30s bucket after five minutes through the original one-hour/
  session terminal. Clock selection uses identifiers and timestamps only.
- Freeze and publish all clock/identity manifests before outcome generation.
- Keep R330's exact 60s hold, BASE −2.5% completed-close stop, three-second
  next-print entry/exit limit and original cost scenarios. Unknowns stay unknown.
- Report every phase and identity, independent economic-target support, original
  observation-weight mass, resolution, signs, costs, stops and clustered support.
  Full unknown means and conditional known means remain distinct.
- Feasibility requires 150 compatible complete episodes, 30 compatible positive
  episodes, both payoff signs on all three dates, and 90% causal-compatible
  clock resolution. Six integrity checks make ten total registered requirements.
- No policy, WAIT continuation, G transport or learner is fitted in this census.
  Later HOLD/EXIT, capital opportunity cost and allocation remain separate work.

## Reviewable implementation

`request331-preregistration.md` fixes population, clocks, economic contract,
denominators, bootstrap seeds, ten gates, reproduction and follow-on decisions.
`request331-inputs.json` freezes 12 existing source/table hashes. Read-only input
authentication passed with exactly 407/828 existing identities; this check did
not construct new clocks or targets.

`src/victory_trader/action_target_support_census.py` implements two separate
clock/outcome stages, atomic tables, early-contract reproduction, all-identity
paired support ledgers and day/ticker-day descriptive intervals.
`tests/test_action_target_support_census.py` has 19 synthetic tests, including
the complete two-stage pipeline, immutable manifests, causal clocks/costs,
unknown executions, censoring weights and independent support counting.
`research/verify_request331.py` compares both JSONs and all eight tables at the
original 1e-9 tolerance, with exact missingness, identities, phase/support flags,
reasons and tie-aware payoff ranks.

`.github/workflows/action-support-census-331.yml` is prepared to use only existing
311/315/319/330 artifacts. It uploads timestamp-only manifests before constructing
outcomes, then preserves the full census artifact. It has not been published or
launched. Inspect commit check-runs before any retrigger to avoid R330's recorded
duplicate-run mistake. Retire it only after verified official closeout.

Critical Ruff and all 19 new tests pass. The full regression suite is recorded
in `request331-validation.json`; the synthetic integration test is not a market
experiment, and no scientific R331 gate result is reported.

## Publication block and exact remaining action

Two attempts to push the local preregistration branch to the same public
`dayofhiki/MoneyMaker` repository were rejected by automatic approval review.
The second attempt followed a Personal Context check of the user's 2026-10-07
standing authorization for same-scope research publication, Draft PRs and cached
official replay. The reviewer still required trusted explicit authorization for
this particular public disclosure of research metadata and dataset-derived hashes.
The remote branch search subsequently returned no matching R331 branch. No
alternate publication path was used.

All unaffected preparation and validation is complete. Remaining blocked scope:
publish the R331 design/code/hash branch to that public repository, create its
stacked Draft PR on the R330 branch, execute one registered cached official
census, reproduce locally from the frozen manifests, and publish honest findings.
This does not authorize a main merge, new market acquisition, opening sealed
dates, changing costs/stops/fill assumptions, promotion or live trading.
