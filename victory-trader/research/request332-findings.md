# R332 complete: missing references are mostly interior gaps; cause unidentified

R332 finished one official cached audit and exact local reproduction. Every
original identity has a retained raw ticker-day path; losing an entire raw file
does not explain the reference bottleneck. Most unresolved compatible clocks
have a later print inside the relevant boundary, but outside the fixed3s limit.
This distinguishes the mechanical gap from file absence, not its market cause.
No return, action-value or account improvement is claimed.

## Exact result, keeping original R331 clocks and execution definitions

| Census | May5–8 training | Primary May6–8 | May11–20 development |
|---|---:|---:|---:|
| Original identities |407|305|828|
| Identities with retained raw ticker-day path |407|305|828|
| No eligible observation inside HOT window |10|5|21|
| EXTENDED clocks |26,433|21,440|52,908|
| Compatible clocks |24,832|20,086|48,061|
| Compatible resolved clocks |7,768|6,997|16,273|
| Pooled compatible resolution |31.2822%|34.8352%|33.8591%|
| Compatible missing entry |12,893|9,812|24,018|
| Compatible missing exit after known entry |4,171|3,277|7,770|

The31 unobserved identities in the full training/evaluation census have raw
records elsewhere in their ticker-day, but none eligible inside their fixed HOT
window. They remain in the denominator. This is not evidence of no trades or a
halt: the normalized seconds have no independent cause record.

| Unresolved compatible reference anatomy | Primary May6–8 | May11–20 |
|---|---:|---:|
| Entry: later print before HOT terminal, lag>3s |9,659|23,577|
| Entry: next print at/after HOT terminal |150|421|
| Entry: no later retained print |3|20|
| Exit: later print before regular close, lag>3s |3,259|7,676|
| Exit: no later retained print |18|94|
| Total unresolved compatible clocks |13,089|31,788|

Thus12,918/13,089=98.6936% of primary unresolved clocks and31,253/31,788=
98.3170% of development unresolved clocks have a late reference before the
relevant boundary. Exit uses regular close, because the original exit contract
can fill after the HOT cap; R332 did not silently change that contract. These are
clock counts, with multiple clocks per identity, not independent order counts.
They are future diagnostics and cannot serve as causal admission filters.

For primary missing-entry clocks the registered gap bins are6,605 in(3,30]s,
1,408 in(30,60]s,1,796 over60s and3 with no later retained reference.
For missing exits they are2,643,318,298 and18 respectively. Boundary categories
and lag bins are different cross-sections; neither changes target resolution.
The complete report includes EARLY, ADDITIONAL_LATE, EXTENDED and every day.

## What the source evidence establishes

Original R311/R315 commits use the same client blob, which requests unadjusted
date-bounded seconds and follows next_url. R311 reports828 calls/cache writes,
zero retries and no failures. R315 reports381 fetched/97 inherited paths in its
original478 union, zero retries and no failures. Those are reported aggregate
counts, not verified page responses for this407 subset. Pair-specific bounds,
adjustment flags, response status, page counts, terminal next_url and payload
digests are absent from the authenticated packages. All remain UNKNOWN.

The fixed packages have no independent full-window trade/quote/minute stream.
HOT minute snapshots and derived features do not provide gap coverage. We did
not open additional dependencies, synthesize an independent comparator from the
same seconds, or call a market API. No-trade, trade-condition exclusion, vendor
loss, halt and actual fill failure remain unseparated. The source audit is
complete; an execution-source readiness claim is not justified.

## Reproduction and publication

Preregistration `e1eb485febcae7b780b844fb9a9a12d9ea8ec270` preceded the census.
Evaluated source `6dbcd3f45eb6763dad979aa8ca4456702ecbc8b7`, tree
`50407e8225e4ba732820f8cf2e9645146b900571`.
Official run[37755971071](https://github.com/dayofhiki/MoneyMaker/actions/runs/37755971071)
succeeded once; no failed/duplicate official audit. Artifact11539558823,
ZIP SHA256`3cd5747f0d8acb8d14566b705c3820ca47f9b2ea4ff23d8039d006090c1caa97`.
Canonical JSON SHA256`480a481c0bd4b90f732e21410d4954080e0a80a56e4a4ff196bc0c7e4d667613`.

All3,814 numeric JSON fields have zero reproduction error; all JSON bytes and
four parquet files are identical. Every category, flag, identity, clock and
missing value matches. Four outputs retain407/828 identities and26,433/52,908
clocks.52 relevant tests passed, including18 new synthetic cases; critical lint
passed. Both full source CI runs37755971037 and37755976171 succeeded; the retained
CI test log reports1,257 passed and160 existing warnings. Full provenance and comparison are in
`request332-provenance.json` and`request332-reproduction.json`.

Draft[PR105](https://github.com/dayofhiki/MoneyMaker/pull/105) is stacked on R331;
no main merge. The one-shot R332 workflow is removed in the closeout commit.
No new market requests, economic labels, fits, promotions or sealed dates.

## Research decision

Keep R331's90% failure and lower memory priority. Another current-state value
fit cannot resolve absent execution observations. Prepare R333 as an independent
source acquisition/reconciliation design for the same fixed identities/windows,
retaining every API page and causal timestamp distinction. Its access/budget
and final scope need to be settled before acquisition. Only then separately
register any new fill/target proxy, followed by chronological ENTER/WAIT account
tests and subsequently HOLD/EXIT/allocation. Do not extend3s, forward-fill trades,
drop unknowns, or choose a contract based on favorable returns.
