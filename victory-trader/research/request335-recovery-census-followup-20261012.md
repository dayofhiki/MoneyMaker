# R335 — retain restored complete chains after a verified storage failure

The independently registered requery stopped after all814 first-four-date chains
were API-terminal:1,481 complete responses,949,890,049 captured bytes, original
407 training identities. A private checkpoint failure stopped new market requests
before the remaining1,656 descriptors. That failed result and its manifest remain
unchanged; it is not called a completed original census.

## Observed storage fault and actual repair

The fourth date archive was reported saved, but its retained/retrieved object was
10,232,717 bytes rather than the expected16,989,354 and could not be parsed to EOF.
Hash verification caught the fault before the next date. The underlying truncation
cause is not established. Local market pages and attempt records remained intact.

An initial version-guarded replacement used the already truncated Library-associated
local archive and did not repair it. A complete archive was rebuilt from the preserved
raw pages under a new path, with an independent held copy. Its guarded replacement
and independent retrieval match SHA256
`08d59cf3f30d43d0d0742df08b2ac172c0435efcf6edd1d3215e48a61e83f847`
(16,989,483 bytes); all537 archived raw/metadata files match the local source.

All four independently retrieved date archives were restored and combined. Raw
page/attempt names are disjoint; the latest final progress snapshot reproduces
the original final manifest bytes exactly. All2,966 restored files match the
original acquisition directory. The frozen full parent raw replay passes:
manifest SHA256
`f60974435ed84ffdbb5225070010728b770b46817488c439a5c6128c20dec4c8`,
replay SHA256
`4c2204b89b8bbf363a14ab87ef19e0e2f2cab6eed93fa1a743fb6f158630f4c0`.
This is actual private-raw restoration, not reconstruction from public counts.

[Observed failure and repair](request335-private-storage-repair-20261012.json).
[Original interrupted ledger](request335-fresh-census-interrupted-ledger-20261012.json.gz).

## Separately registered completion using complete-chain evidence

[New registration](request335-recovery-census-registration-20261012.json) and
[contract](request335-recovery-census-contract-20261012.json) precede any request
for the remaining original descriptors. A fresh destination copies and verifies
the restored parent evidence. Only its814 complete original query chains are
reused, by exact descriptor/index. No failed/partial query chain, page cursor or
economic outcome is reused; the earlier stopped process is not restarted or its
failure cleared. All1,235 identities /2,470 descriptors remain in the new ledger.

The previously executed parent raw replay is hash-pinned as a cache. Before reuse,
the parent manifest, original descriptors, every raw body/attempt file, contracts,
rights and frozen source hashes are verified again. The cache cannot replace
missing raw files or be trusted after a byte/source change. It avoids repeatedly
decoding the same restored rows while retaining actual current raw authentication.
The eight previously unrequested dates begin at page1 with independent provenance.
Reuse is acquisition evidence; it does not establish one provider revision epoch,
historical arrival order, source qualification or executable quotes.

Dates already restored retain their exact private archive proofs. Each new date
still pauses for private saving/retrieval/raw-file comparison before the next.
The controller holds an independent immutable archive before sending a working
copy. If a known successful save yields different bytes, it permits one guarded
repair of that same private item from the held master, then re-verifies. Unknown
write outcomes are not retried. Failure to establish the complete bytes stops new
market requests. HTTP/transport/schema/sensitive/repeated-token errors retain the
frozen stop/no-retry/no-fallback policy and original fixed budget/window controls.

## Validation and scope

Four added synthetic tests cover all814 reused /1,656 new original chains,
different-date request isolation, new403 stopping without dropping inherited or
unrequested identities, changed parent raw bytes and false incomplete-chain reuse.
Critical Ruff passes; all1,597 local tests pass (Python3.12.14,243 warnings,81.74s).
The prior independent source CI also passes1,593 tests (160 warnings,140.92s).

No source or economic gate is relaxed: source_ready=false, economic labels0,
fits0, profitability incomplete, R331 pooled34.8352% failed,90% resolution,
all-phase/all-clock denominator,60s hold,3s boundary and BASE−2.5% completed-close
stop unchanged. June HOLD /July·August SEALED. Incremental spendingUSD0; no account,
plan, order, support message or main merge. Raw pages/tokens/private IDs remain
outside public repo/CI. Actual completion and final private restoration results
will be recorded below when verified.
