# R335 preparatory stage: offline source-input inventory

This stage continues R334 without assuming a new entitlement or buying data.
It implements an offline inventory of explicitly supplied, already authorized
exports. It is not the unexecuted R335 acquisition/reconciliation experiment.
No market request, credential access, account change, economic label, fit or
sealed date is part of this stage. Existing standing research approval covers
this preparation; a new paid subscription still requires an exact purchase choice.

## Frozen scope and limits

Authenticate the original R332 source SHA and both source-ledger bytes against
R333's committed input hashes. Rebuild all 407 training / 828 reused evaluation
identity windows, using the same session calendar and max(open,HOT-60s) to close
contract. Keep the primary May6–8 identities separate. Retain both required tick
streams for every identity, including unsupplied, partial and invalid entries.
The 12 R333 qualification anchors are never a substitute for the full census.

An explicit source catalog maps a ticker/day/stream to an original REST-row JSONL
or provider CSV file plus an independently hashed provenance file. No directory
scan or credential search. Paths must stay within the supplied export root;
absolute paths, symlinks and parent traversal are rejected. Maximum 32 MiB per
file, 256 MiB total including provenance, and 1 MiB per record. These are offline
inventory ceilings, not permission to download this amount. Larger daily files
need a separately budgeted import design; do not silently sample or truncate.

For this first adapter all event clocks are explicitly nanoseconds. JSON clocks
must be integers (not bool/float); CSV clocks must be decimal integer strings.
Require SIP clock, count missing participant/TRF clocks and conditions, retain
duplicates, cancellations, out-of-order records and original fields. Do not
deduplicate, correct, infer units, filter eligibility, forward-fill or price a
trade. File hashes cover original bytes, not a reconstructed normalized stream.

Declared bounds must remain on the original May regular session and cover the
identity window. A matching hash establishes byte identity, not origin truth,
licensed access, provider page completeness or causal strategy availability.
Every supplied file remains SOURCE_COMPLETENESS_UNVERIFIED until a separately
registered replay authenticates the acquisition/page-chain evidence. A catalog
`complete=true` or first/last record cannot establish completeness. Empty files
are an explicit observed empty export, not evidence of no trades.

## Verification and decision

Synthetic tests must exercise precision, half-open bounds, scope, hash mutation,
partial/missing exports, duplicates, quote fields, path confinement, immutable
output, unsupported formats and budgets. Run the actual original census with an
empty catalog and verify that all 2,470 identity/stream records remain unknown.
Run it twice independently and compare exact report/ledger bytes. This is an
offline readiness result, never independent market evidence or an official
source-acquisition run. No one-shot market workflow is added.

Only after an existing export or verified account entitlement is supplied can
the acquisition/import contract advance. R331's 3s execution bound, 34.8352%
primary resolution and failed 90% gate are unchanged. The next economic stage
still requires a causal order/quote proxy before continuous ENTER/WAIT and then
HOLD/EXIT/allocation learning.
