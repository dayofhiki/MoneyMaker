# R335 — fixed anchors, full-window technical feasibility

2026-10-11: a separately registered independent repeat completed all24 selected
API chains at additionalUSD0 and verified original prefixes and private restore.
See [the latest repeat and qualification report](request335-recovery-followup-20261011.md).
This document preserves the earlier interrupted acquisition, including its failure.

**실제 후속 결과:** 무료 전체 구간 기술 수집을 진행했다. 14회 요청 중 13개
HTTP 200 응답에서 체결 12,769건 / 호가 17,825건, 3,209,029바이트를 보존했다.
24개 표본 구간 중 12개에서 API 종료를 확인했으며, 14번째 요청의 전송 오류로
계약에 따라 중단했다. 실행 도구는 네트워크 승인이 결정 전에 취소되었다고
반환했다. 추가 비용 $0, 새 경제 라벨 0, 수익성 검증 미완료다.

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

## Actual collection and independent verification

Source and contract were published **before** collection in commit
`40823835ea36ea4c08707acd71562d2e11f8561f`, tree
`c61d0edfd528eb0351dba985ebfefaedd7554416`. Collection ran from
2026-10-10 16:30:20.773671 UTC through 16:31:56.698820 UTC, approximately
95.92 seconds. The collector source has not changed since registration.

| Evidence | Actual result |
|---|---:|
| Selected original full-window partitions | 24 |
| HTTP attempts | 14 |
| Complete HTTP 200 pages | 13 |
| Attempt without an HTTP response | 1 |
| Retained response bytes | 3,209,029 |
| Trades / quotes observed | 12,769 / 17,825 |
| Terminal API chains | 12 |
| Selected identities with both API chains terminal | 5 of 12 |
| Scientifically qualified market partitions | 0 |
| Original full-census partitions not requested | 2,456 of 2,470 |
| Additional spending / economic labels / model fits / orders | $0 / 0 / 0 / 0 |

The failure was the AARD quote request on 2026-05-13. Its HTTP status is absent,
retained body is zero bytes, and no observation is admitted. The collector
records `TRANSPORT_ERROR`; the execution tool separately returned
`network approval was cancelled before a decision was returned`. These records
do not establish provider denial, a subscription limit, expired credentials,
rate limiting or a server outage. The earlier 13 HTTP 200 responses remain
valid evidence of successful access. No retry, new acquisition, alternate route,
login, subscription change or payment was attempted after this stop.

The AIRS quote chain has one retained page and a continuation token; it is still
incomplete. Ten later selected partitions were never requested. Their entries
and the AIRS entry remain `INCOMPLETE_AFTER_TRANSPORT_ERROR`; AARD quotes remain
`TRANSPORT_ERROR`. Combining the pinned original full plan with this exact
selected-pair ledger preserves all 2,470 original partitions, including every
unrequested identity. Unknown observations receive no no-trade, cash or loss
label. The earlier 60-second and AIRS page-check acquisitions remain unchanged.

Two canonical offline full-window replays match byte for byte. Private manifest
SHA-256 is `334a6ef53cfa9269deacdfda2481442b5f7f36dba9a7dc2071bcbe33ac49047c`;
public replay SHA-256 is
`d8fe4fac46d091fbcd8c5943b7f02a127c3c3de67966b35296873f3a4c271feb`.
Directory and file permissions remain 0700 / 0600. No keys, authentication
headers, account identifiers, raw ticks or pagination tokens are published.

The separate offline semantics audit and original-prefix comparison also each
match on two runs. All 13 pairs with successful fresh pages match the retained
original first-minute rows in their original order, with no changed field at
those ordinals. Eleven pairs have no successful fresh page and are not
comparable. This checks the observed prefixes across query limits/windows;
it does not prove the entire historical record or correction delivery history.

Required-field/schema errors, observed backwards clocks and identical decoded
rows are zero. All 17,825 quote quantities map to shares under the official
dated unit rule. There are 763 locked and 12 crossed quotes, retained verbatim;
their eligibility for execution remains unqualified. All 12,769 trade rows lack
optional update field `u`, and their historical size unit is still unverified.
Absence of `u` does not establish absence of corrections. This subset has no
observed equal-clock groups; the equal-clock groups in the earlier full 12-day
first-minute sample remain recorded. No receipt clock or event sequence was
invented, and no source diagnostic confirms a fill.

Source CI also passed: push run 38067883371 / test job 114259149217 and PR run
38067886329 / test job 114259157299 each pass critical lint and all 1,525 tests
(Python 3.11, 160 warnings; 72.01s / 85.87s). Credentialed metadata jobs were
skipped. The result commit adds an offline audit/comparison wrapper and records;
application collector and tests are unchanged from the validated source.

## Economical full-census path and remaining conditions

The recommended acquisition route remains the attested **free Basic historical
SIP** route for permitted private personal research. Actual successful full-window
requests now demonstrate more than first-minute access. Execution-environment
network access needs to be permitted again before further collection. Current
evidence does not call for an Alpaca account operation or payment.

| Path | Cost / billing basis | Remaining technical limitation |
|---|---:|---|
| Existing licensed terminal raw chains | $0 | 12 candidate chains; source qualification and unchanged raw provenance still required |
| Basic historical SIP for remaining original windows | $0 under the attested account scope | At least 2,458 fresh first-page requests if those 12 chains can be reused; pagination tail, omitted events, OTC and chronology remain unknown |
| Current Massive Starter | $0 above the existing $29/month | Required trades/NBBO remain unavailable; aggregates cannot replace them |
| Massive Advanced | $199/month list price | Requires advance approval; source/correction/clock/retention qualification still applies |
| Academic TAQ / separately licensed samples or subsets | Entitlement or quote dependent | Exact modules, May 2026 coverage, legal use and retention must be verified |

The full vendor, price, precision, coverage, rights and reproducibility comparison
remains in [the cost review](request335-cost-integrity-review.md). No alternate
partial-exchange BBO is treated as official SIP NBBO. No paid sample or API was
used. A support letter is not a prerequisite for already documented free access.

`request335-full-census-budget-20261010.json` is a **budget assessment**, not a
full-census collection registration. A completely fresh census requires at
least 2,470 first-page requests: request-start spacing alone is at least 2,469
seconds at one second per request. Reusing 12 terminal raw chains conditionally
reduces this lower bound to 2,458 requests. Neither bound includes extra pages,
response time or source failures. The nonrandom pilot is incomplete and
transport-censored, so no population mean, expected bytes, completion time or
confidence interval is inferred from it.

A hypothetical 32-page stop for every partition gives a finite resource ceiling
of 79,040 attempts / 154.375 GiB at 2 MiB per response, with at least 79,039
seconds of request-start spacing. That is a cap scenario, **not** an upper bound
on the data required: denser streams could still be censored. The pilot's
256-attempt / 30-minute budget cannot authorize or complete the full census.
A separate fixed-census, shard, page, byte, disk and wall budget must be
registered before that acquisition; all failures and caps remain in its ledger.

The minimum conditions for the next stages remain:

1. For another free acquisition: permitted API connectivity, the existing
   free-account/license scope, an independent registered scope/resource contract
   and immutable previous evidence. The interrupted run is preserved.
2. For full source qualification: every original identity's full trades/NBBO
   windows and page-chain provenance, exact nanoseconds and documented clock
   meaning, share units, quote conditions, halt/OTC coverage, corrections and
   equal-clock order, with omissions explicit. Terminal API markers alone do not
   satisfy market completeness or the original resolution criterion.
3. For an execution experiment: register causal availability/latency and quote
   age, executable sides/size/participation, order/cancel/partial-fill rules,
   slippage and fees, stop/censoring behavior, chronological capital use and
   opportunity cost. Synthetic data remains code validation only.
4. For economic conclusions: preserve 407 training / 828 reused-development
   identities, the original 90% resolution criterion, 60-second hold, 3-second
   boundary and −2.5% BASE completed-close stop. June remains HOLD and July/August
   sealed. R331's 34.8352% pooled resolution stays a failed gate. No profit claim
   or replacement evaluation is enabled by this pilot.

Offline audit and prefix reproduction (requires the private evidence, no API):

```sh
python research/verify_request335_full_window.py \
  --evidence /path/to/full-window-technical-20261010 \
  --original-evidence /path/to/technical-sample-20261010 \
  --output /new/path/audit.json --prefix-output /new/path/prefix.json
```

Public evidence: `request335-full-window-result-20261010.json`,
`request335-full-window-audit-20261010.json`,
`request335-full-window-prefix-comparison-20261010.json`,
`request335-full-census-budget-20261010.json`, and
`request335-full-window-validation-20261010.json`.
