# R335 — fixed anchors, full-window technical feasibility

This increment is acquisition and replay research. It does not enable economic
labels, model fitting, execution claims, or a change to the original experiment.
The previous 60-second evidence and separate AIRS page check remain immutable.

## Registration before market requests

`request335-full-window-contract.json` fixes the next collection. The pinned
source plan remains `request335-free-source-plan.json.gz`, decompressed SHA-256
`0ad2603a2a62ef49a975f21c5951e55fa6708fbe0d062bf5df5c832bb88c235e`.
Selection is the same lexical-first anchor on each of the twelve original May
dates, with both trades and quotes: 24 full-census descriptors. Each window is
`[max(regular_open, HOT−60 seconds), regular_close)`. The HTTP inclusive end is
exactly the original exclusive end minus one nanosecond. No June, July, August,
symbol remapping, alternate feed, or favorable-price selection is enabled.

Round-robin scheduling requests one page from each active partition before its
next page. All 24 entries stay in the ledger. Terminal responses prove the API
page chain ended, not that the market stream is complete or a trade executable.
Other original partitions stay explicitly unrequested; the denominator remains
1,235 identities / 2,470 identity-stream partitions.

| Resource | Hard stop |
|---|---:|
| HTTP attempts | 256 |
| Pages per partition | 32 |
| Retained bytes per response | 2 MiB |
| Total retained response bytes | 256 MiB |
| Request start interval | at least 1 second |
| HTTP timeout | 30 seconds |
| Collection wall time | 1,800 seconds |
| Free disk floor, plus space for one response | 512 MiB + 2 MiB |
| Retries, redirects, automatic resume, alternate feed | 0 |
| Additional authorized spending | USD 0 |

These caps are resource limits, never success thresholds. A partial page, cap,
repeated token, invalid schema or denied partition remains incomplete. Any HTTP,
transport, schema or sensitive-body error stops the run; a per-partition page cap
retains that partition and allows the other fixed partitions to continue.
Retained-byte limits do not claim to meter remote network traffic: a streamed
chunk can be received before the retained prefix reaches the limit.

## Access and rights

Reuse the already authorized private Alpaca Paper credentials and free Basic
attestation. This registration extends the technical sample from the first
minute to the original full windows of the same twelve anchors. No subscription,
purchase, order, external message, GitHub secret, or credential destination is
added. Requests go only to HTTPS `data.alpaca.markets`; headers and pagination
tokens stay private. Raw wire files and attestation are retained locally with
0700 directories / 0600 files. Public artifacts contain counts and hashes only.

Official rules permitting historical SIP on Basic, the price comparison and
rights limits remain recorded in `request335-semantics-followup-20261010.md` and
`request335-cost-integrity-review.md`. The 15-minute recent-data restriction
cannot affect these May windows. Private personal research and local retention
are the attested scope; redistribution and commercial/perpetual rights are not
inferred. No vendor support letter is a prerequisite or has been sent.

## What this can establish

The collector preserves raw decimal values and RFC3339 timestamps to integer
nanoseconds, checks exact original bounds, required field presence, page tokens,
hashes, row order and response budgets. Private checkpoints allow inspection of
a crash without silently resuming. Offline replay checks the registered source,
implementation, attestation, all original selected descriptors, round-robin
order, transport rate, exact page bytes, token chain and final states.

May quote quantities are shares under the official 2025-11-03 unit change;
the existing adapter already records that dated rule. Quote zeros can mean an
inactive side and are counted rather than imputed or discarded. Historical trade
quantity units, quote condition eligibility, equal-clock sequence, correction
delivery, clock semantics and vendor omissions still require qualification.
The provider timestamp is never silently relabeled as a SIP or receipt clock.

Full-window pilot throughput can inform a separately registered full-census
budget, but its lexical sample is not representative of density, OTC access,
missingness or profitability. A terminal pilot cannot establish the success of
the full R335 experiment. The original 90% resolution gate, all-phase/all-clock
denominator, 60-second hold, 3-second boundary and −2.5% BASE completed-close stop
remain unchanged. R331's failed pooled-resolution result stays failed.

## Reproduction

Run the synthetic tests without credentials:

```sh
python -m pytest -q tests/test_r335_full_window_qualification.py
python -m ruff check --select E4,E7,E9,F src tests
```

The production CLI accepts only the pinned plan and approved small private
credential and attestation files. `contract` and `replay` make no network calls.
`collect` requires a fresh private output directory; no automatic resumption or
paid fallback exists. Publish this source and contract before executing it.

Actual collection, replay results, validation hashes and the full-census budget
assessment will be appended in a separate result commit after registration.

Local source validation before registration: 33 new synthetic cases pass; all
1,525 tests pass (243 warnings, Python 3.12.14, 291.61 seconds with numeric
thread limits set to one). Critical Ruff E4/E7/E9/F passes. An initial default
thread run was interrupted for resource contention without reported test
failures; its partial execution is not counted as a pass. Both prior private
replay hashes still match their published records.
