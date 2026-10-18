# R335 — finish the unchanged census with atomic private archive publication

The next acquisition reuses2,052 complete original chains authenticated against
their actual raw/attempt files and frozen replay caches. It starts only the418
previously unrequested May19/20 descriptors at page1. All1,235 identities (407
training /828 reused development) and2,470 stream descriptors remain explicit.
No partial query, page cursor, missing outcome or failed history is stitched in.

## Second observed checkpoint fault and actual private repair

The last recovery stopped after2,312 new HTTP responses plus1,481 inherited
responses. All2,052 inherited/new complete chains remain intact. The May18
archive's readiness record specified23,618,893 bytes and SHA256
`938e032dbc9a2f418afe2c60ad10b564d7f590940364dd293d2379855e2d83fe`;
the subsequently observed source/held files were23,048,761 bytes with a different
hash. This mismatch was caught before any upload was attempted. Its underlying
cause is not established. It is an archive handoff observation, not a market
data HTTP failure. The stopped manifest and all earlier failures are preserved.

A new canonical atomic archive was created from the retained pages, saved to the
user's private store, independently retrieved and restored. Its23,574,342 bytes
match SHA256`5df975c2477962440508828f27893d07f3364637ec6a207011191eb0ade7c40e`;
all627 archived files match the source. The current full raw replay also passes:
manifest SHA256`667878ad607dc203e6df5b51abe25aafd89822d92f0a9c5c9044884afd233eca`,
replay SHA256`7867dbfe6a7fd0c009f13e5fe75aad6d4e9e0147673e29af27749b642f2eaa8e`.

[Fault/repair evidence](request335-atomic-storage-repair-20261012.json).
[Preserved interrupted ledger](request335-recovery-census-interrupted-ledger-20261012.json.gz).

## Atomic publication and bounded readiness

The new archiver writes a fresh temporary file, closes the gzip/tar stream,
flushes/fsyncs it and computes its hash/size before atomically renaming the final
archive; the containing directory is fsynced too. Gzip filename/time and external
tar ownership/mode/time headers are canonicalized. Original market bytes and
acquisition timestamps inside JSON evidence remain unchanged. The same paused
source files produce identical archive bytes across output names and filesystem
mtime changes.

Before upload, the controller waits read-only for the exact expected hash and
size. If necessary it may recreate the same canonical archive from paused source
files in a separate foreground path; that archive must match the expected hash
exactly before any upload. An independent held master is retained. Actual saved
bytes are independently retrieved, checked and raw-restored. A known successful
but corrupt write permits one version-guarded repair from the master; an unknown
write outcome is not retried. The existing600s confirmation timeout and stop
policy still prevent the next date from starting without a verified checkpoint.

Ten private date archives are already verified, including post-stop repairs.
The last two receive the same private checkpoint and final full-archive checks.
The previous stopped process is not resumed or its status cleared. This is a
separately registered completion using full original-chain provenance.

## Registered scope and validation

[Registration](request335-final-census-registration-20261012.json) and
[contract](request335-final-census-contract-20261012.json) pin sources, dependencies,
original descriptor plan, parent manifest/replay and private ten-date proofs.
Publish before the first remaining request. The frozen SIP/asof-/USD windows,
request/response/page/rate/concurrency/wall/disk controls remain unchanged.
There is no paid route, API error retry, redirect, alternate feed/account or
automatic restart. Previous acquisition epochs retain separate provenance and
are not represented as one frozen provider revision or historical receipt order.

Two new atomic-archive tests prove no final file is visible before closing/fsync
and demonstrate identical archive bytes across filename/mtime changes with exact
raw restoration. Two nested-reuse regressions prove2,052 complete chains are
reused and only418 original descriptors queried, and that a missing inherited
raw file cannot be replaced by a published cache. Critical Ruff passes.
Full-suite results are recorded in the validation file and PR.

source_ready=false /qualified0 /economic labels0 /fit0 /profitability incomplete.
Original90% execution resolution, all-phase/all-clock denominator,60s hold,3s
boundary, BASE−2.5% completed-close stop and R331 pooled34.8352% failure are retained.
June HOLD /July·August SEALED. Incremental spendingUSD0. No account/plan/order,
support message or main merge. Native private IDs and market rows stay outside
public repo and CI. Actual final acquisition/retention results follow when verified.
