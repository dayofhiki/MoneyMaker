# R332: fixed cached source and missing-reference audit

Registered after R331 closeout `3068933716011dc8bb7549b8b3d55d1bbd4bde74`.
Publish this contract and input hashes remotely before running the new census.
The prior proposal is superseded by this executable scope. This is descriptive
source diagnosis, not a new target, policy, learning experiment or return claim.

## Authenticated scope

Use only original R311, R315, R319 and R331 artifacts listed in
`request332-inputs.json`. Fixed identities: all 407 broad May5–8 training
identities; primary 305 May6–8 separately; all 828 May11–20 development identities.
Retain unobserved identities and ALL registered R331 EARLY/ADDITIONAL_LATE clocks;
EXTENDED is their union. No June HOLD, July–August, new acquisition, credentials,
fits, action eligibility changes, economic label computation or account replay.
Read raw timestamps and existing R331 timing/reason/compatibility flags only;
do not read prices or returns for the census. Full parquet digests authenticate
the bytes without using those values. No threshold search or bootstrap.

The available-source inventory was checked before registration: the authenticated
artifact packages retain normalized seconds, derived states/cohort snapshots,
aggregate acquisition summaries and clocks/targets. They do not retain raw API
page payloads/request-response manifests or independent full-window minute,
trade or quote streams. This is a scoped absence in these packages, not a claim
that no other historical artifact or provider record exists. Do not download
additional historical dependencies or call a market API in this stage.

Original client blob `032604d836e381e2bfd18fbd1b673065cd076a0c` is identical
at the R311 and R315 source commits. It requests date-bounded unadjusted seconds
with ascending order/limit50000 and follows next_url. The original workflow
uploads artifacts/**, not data/cache/massive-second-attention. Code evidence and
aggregate successful-call counts cannot prove pair-specific payload/page-chain
completeness. The source-evidence JSON records code facts and exact provenance
unknowns without URLs containing credentials or raw responses.

## Fixed computations and semantics

1. Verify every input hash and original source SHA. Copy every identity and
   registered clock unchanged; exact one-to-one join existing timing fields.
2. For each identity, count full regular-session cached seconds and seconds
   whose completed interval is eligible within [HOT, min(HOT+3600s, session close)).
   Record first/last cached starts, before-HOT/after-terminal counts, observed
   clock counts and no-observation status. Retain zero-row/missing pairs.
   A last print before session close does not imply truncated acquisition.
3. Independently locate the next cached start at/after each decision and each
   retained exit_submission_t, using timestamps alone. Check the recorded R331
   next-reference gaps including NaNs exactly. No exit submission is recomputed.
   Bins in seconds: <=3, (3,30], (30,60], >60, no later cached reference.
4. Entry anatomy distinguishes within3s before terminal, late before terminal,
   later reference at/after terminal, no later cached reference, and raw pair
   absent. Exit anatomy uses within3s, late within session, at/after session close,
   no later cached reference and raw pair absent. Exit may legitimately occur
   after HOT terminal under the original contract; do not silently cap it at HOT.
   These are retrospective source diagnostics, unavailable to action selection.
5. Summarize all clocks and compatible clocks for all three arms and three
   scopes, per day and pooled. Separately summarize missing-entry clocks and
   missing-exit clocks after a known entry. Keep compatible unresolved and known
   counts; reproduce R331's pooled resolution, never replace its 90% failed gate.
   Mechanical later-print categories do not establish economic fillability.
6. Inventory retained lineage evidence by source: original SHA/blob and input
   digests authenticated; acquisition failures/client counts as REPORTED;
   actual per-pair request bounds/adjustment/status/page count/terminal next_url/
   response digest UNKNOWN. R315 inherited97/fetched381 cover the original478
   union, not this407 subset; no unsupported per-pair attribution.
   Independent same-window streams: NONE RETAINED in the fixed input packages.
   Cause attribution remains UNIDENTIFIED for unresolved references and empty
   windows: no-trade, ineligible trades, halt, vendor loss and true failed fills
   cannot be separated with these inputs. Do not synthesize an independent
   comparator by reaggregating the same seconds or use HOT minute snapshots as
   full-window evidence.

## Validation, official reproduction and decision

Synthetic tests cover bin boundaries, missing pairs, terminal/session distinctions,
immutable clock joins and the distinction between report/code facts and actual
response evidence. Run critical lint and relevant tests; one independent cached
GitHub workflow downloads only the four original artifacts. Publish source SHA,
environment, report, complete identity ledger and complete clock audit. Compare
local and official JSON numerics (tolerance1e-9), categorical values, missingness,
identity/clock sets, parquet values and byte digests. Preserve failed attempts;
do not silently rerun or select a favorable replica. Retire the one-shot workflow
after verified closeout.

The audit succeeds operationally when the fixed inventory and census reproduce;
source completeness and gap causes can remain unknown. No source-readiness pass
may be manufactured from missing evidence. If unknowns persist, freeze a separate
acquisition proposal specifying exact identity/windows, page/digest/availability
manifest and independent eligible-trade/NBBO reconciliation before any new order
proxy or target. Do not weaken lag/cost/stop contracts or start a value fit merely
because additional clocks exist. Standing routine cached-research publication
approval applies; new market collection remains outside this stage.
