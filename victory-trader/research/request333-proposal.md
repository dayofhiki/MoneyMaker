# R333 proposal: independently observed source gaps before new action targets

Prepared after exactly reproduced R332. Not preregistered, collected or executed.
R332 identified interior late references, not causes:98.7% of primary unresolved
compatible clocks have a later print before the relevant boundary. Every raw
ticker-day path is present, but response/page completeness and independent market
records are absent. Improve this observation contract before another value fit.

## Fixed census and collection boundary proposed

All407 original broad May5–8 identities and828 original May11–20 development
identities; primary305 May6–8 reported separately. Use authenticated original
identity and R331 clock files. Retain unobserved and unresolved identities; no
payoff/phase-based sampling. No new dates, June HOLD or July–August.

For each ticker-day identity propose `[max(regular_open,HOT-60s),regular_close)`
as the independently collected window. This session-tail endpoint is deliberate:
R331's stored next-reference diagnostic searches the entire retained session,
and a fixed exit at the HOT cap can submit at that boundary. A shorter window
would introduce new censoring. Same ticker-day overlapping windows may be merged
for transport, but each identity retains its original window. Include60s before
HOT for timestamp alignment; do not infer an older quote that was never retained.
If transfer/access costs require a smaller design, preregister a replacement
before requests; do not shrink windows after seeing prices or coverage.

Required source bundle: independently downloaded unadjusted second aggregates,
minute aggregates, full trade records and historical NBBO updates for each window.
Independent here means separately retained source responses, not different vendor
ground truth. Minute/trade eligibility must be evaluated with a pinned provider
condition specification; minutes do not prove executions. Historical quotes
provide a market observation proxy, not an actual receipt, queue position or depth.

## Evidence that must survive normalization

One immutable record for every requested identity/window/stream, including empty,
failed, unauthorized and partial responses. Record endpoint/parameters without
credentials, request/response UTC, HTTP/provider status and request ID, response
adjustment flag where defined, query/results counts, actual first/last event time,
raw page SHA256, page number, sanitized next-page linkage and an explicit terminal
page marker. Retain raw page bytes safely and a canonical normalized digest.
Persist missing optional fields as missing, not successful defaults. Manifest
upload/checkpoint precedes reconciliation or any future targets. Failed/partial
page chains remain unknown; reported call count cannot replace this evidence.

Keep SIP and participant timestamps, sequence/identity, trade conditions and
correction indicators when provided; preserve quote bid/ask, sizes, exchanges,
conditions and timestamps. Use integer nanoseconds; no float conversion. Preserve
duplicates, corrections and timestamp disagreements in an audit before defining
deduplication. Pin the eligibility/correction specification and exact availability
rule before reconciliation. SIP event time is not a measured strategy receive
time; historical final corrections may not have been known at the original clock.
Do not silently label a final corrected historical stream as an online feed.

## Registered reconciliation proposed, without economic targets

Join all R331 entry decisions and existing exit submissions against independently
retained streams, including each empty HOT window. Fixed bins remain3/30/60s.
Report original-versus-new aggregate starts, missing/extra seconds, and byte/value
differences separately; a later acquisition may contain corrections. Full-window
trade coverage distinguishes no returned trades from returned-but-ineligible
trades only where page completeness and the condition specification support it.
Trade-reconstructed aggregates are a reconciliation of the new independent tape,
not proof that the old acquisition was wrong at its original collection time.
Report qualifying/ineligible/unknown-condition counts, response completeness,
quote presence/staleness, size/spread validity, crossed/locked markets and
SIP/participant timing conflicts. Missing records stay unknown. Halt attribution
requires a separately pinned status source; do not infer a halt from gaps.

Success means reproducible page-complete source evidence and a transparent
classification of identifiable versus unknown gaps across the full census. There
is no90% source gate chosen to rescue R331. No model, causal selector, stop,
holding-horizon or payoff calculation in this stage. Preserve old3s result.

## Conditions before execution and follow-on work

Confirm the user's intended existing market-data account, historical stream
entitlements, authorized request/byte budget, rate limits and durable raw-page
destination through an explicit source contract. No credential probing, account
purchase, new collection or quote access is authorized by this cached proposal.
Prepare and remotely register a final window/eligibility/correction/manifest
contract after access is known; dry-run transport with synthetic pages first.

A later separately registered NBBO/order proxy must specify causal receipt-time
assumptions, quote age/size and latency, spread/slippage/fees, stops, cancellation,
terminal/unresolved capital, chronological validation and account drawdown. Its
thresholds come from order mechanics and source support, not profitable returns.
Do not carry a stale trade as a synthetic fill or erase missing orders. Only with
adequate economic targets should the continuous current-state ENTER/WAIT learner
advance; learned HOLD/EXIT and ADD/REDUCE allocation remain subsequent work.

Primary provider documentation checked2026-10-08:
- https://www.massive.com/docs/rest/stocks/aggregates/custom-bars
- https://www.massive.com/docs/rest/stocks/trades-quotes/trades
- https://www.massive.com/docs/rest/stocks/trades-quotes/quotes
These describe source fields, not the user's entitlements or guaranteed fills.
