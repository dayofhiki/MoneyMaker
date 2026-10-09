# R333: bounded historical-source qualification before full reconciliation

This final stage contract supersedes the collection portion of the R333 proposal.
Publish this contract, input hashes and deterministic plan before any requests.
Continuing the user's research uses the existing MoneyMaker Massive account via
its configured Actions secret only. No new account, credential search, purchase,
subscription change, sealed date or live order. This is a source qualification
stage; the full1235-identity acquisition/reconciliation remains separate R334.

## Fixed identities, timestamp-only selection and windows

Authenticate R332's original source-ledgers407/828 and source SHA by
`request333-inputs.json`; preserve all1235 identities in a full-window manifest.
Full design window: [max(regular_open,HOT-60s), regular_close). Its SHA256 is
registered in `request333-plan.json`. Do not collect those full windows here.

For qualification, select lexicographically first(day,ticker,HOT) identity per
original day,12 identities total. No prices, returns, missingness or observed
quote availability affect selection. Qualification window is the FIRST60s of
each declared full window, capped at regular close. Preserve all48 declared
identity/stream pairs, including denied, failed, partial and unrequested ones.
Streams in fixed order: seconds, minutes, trades, quotes, for each sorted day.
This sample establishes access/transport facts only; it cannot estimate full
census support, economic returns or gap causes. Do not relabel R331 unknowns.

## Source and hard resource limits

Existing configured `MASSIVE_API_KEY`, HTTPS `api.massive.com` only. Aggregate
endpoints use multiplier1, second/minute, adjusted=false, sort=asc, limit50000,
from=start_ms/to=end_ms-1. Tick endpoints use timestamp.gte=start_ms*1000000,
timestamp.lt=end_ms*1000000, sort=timestamp, order=asc, limit100. Preserve exact
integer SIP/participant nanoseconds; never pass timestamps through float.

Maximum96 HTTP attempts,2 pages per declared pair,2MiB per body and32MiB total
retained response bytes, at least12.5s between request starts, timeout30s. No
automatic retries. 401 stops ALL remaining network requests. 403 stops later
requests of that stream; unrequested pairs are explicitly SKIPPED_AFTER_DENIAL,
not silently declared denied. 429 stops ALL remaining requests as rate-limited.
Other HTTP/provider errors, invalid JSON/schema, missing tick timestamps,
unsafe/repeated pagination, byte/request limits and partial pages are retained
with explicit non-complete status. No larger window, higher budget, different
identity or alternate account is tried to obtain a successful result.

Credentials absent: record every pair as NOT_REQUESTED_MISSING_CREDENTIAL,
without probing environment, files or account settings. API access is measured
by the responses, not inferred from public plan documentation or old S3 access.
A zero-row successful complete page means this endpoint returned no records in
this sampled window; it is not proof of no market trades/quotes.

## Evidence, secrecy and exact replay

For each attempted page retain sanitized request path/parameters, page number,
request/response UTC, HTTP/provider status, request ID, counts, response adjusted
flag (or missing), terminal next_url marker and sanitized pagination linkage.
Record SHA256 of received wire bytes. Preserve wire bytes only when they contain
no credentials/sensitive keys or credential-bearing links. Otherwise withhold
wire bytes, retain sanitized canonical JSON and mark the wire evidence withheld.
Do not falsely claim the sanitized bytes equal the original response. Partial
oversized bodies retain only bounded bytes with a prefix-digest flag. Exceptions
are recorded by type only; do not print URLs, headers, secrets or body messages.
Validate HTTP success AND provider success AND proper result schema before
granting observed page completeness. next_url is restricted to the exact same
endpoint/host; remove all credential parameters before the next request.

Upload the immutable page/manifest artifact before analysis. Offline analysis
authenticates every retained body and reconstructs stream/pair state, rows,
timestamp ranges, out-of-window/order/duplicate diagnostics and optional-field
missingness. No tick deduplication/correction/condition filtering or quote-age
choice in R333. Count returned corrections/conditions and missing SIP/participant
timestamps without calling them eligible, causal or filled. Quotes retain raw
prices/sizes/conditions as source data, not economic targets. Historical final
records and SIP times do not establish strategy receive times. Eligibility and
availability rules require a later preregistration before any reconciliation
classification or new order/target proxy.

One official qualification run; preserve failed attempts and no silent rerun.
No second market acquisition for reproduction: authenticate and replay those
same archived pages locally, comparing all report numbers (tolerance1e-9),
categories, missingness, identities and immutable manifest bytes. Independent
transport tests use synthetic HTTP pages, including denied/partial/malicious
pagination, budgets, nanoseconds, secret redaction, empty pages and error states.
Retire the one-shot workflow at verified closeout.

## Decision

Report exact sampled facts for each stream. If quotes/trades are denied or
unverified, do not collect a misleading full comparator or refit current-state
values. Prepare the concrete missing-access choice using observed statuses; do
not buy access. If all required streams are supported, preregister R334's exact
full window/page budget and provider eligibility/correction specification before
the full1235 census. R331's34.8352% primary resolution and90% failure stay intact.
R333 never claims that transport qualification improves profit or identifies
the old gaps. June HOLD, July–August, costs, stops and policy status are preserved.

Provider specifications checked2026-10-09 (not account entitlement evidence):
- https://www.massive.com/docs/rest/stocks/aggregates/custom-bars
- https://www.massive.com/docs/rest/stocks/trades-quotes/trades
- https://www.massive.com/docs/rest/stocks/trades-quotes/quotes
