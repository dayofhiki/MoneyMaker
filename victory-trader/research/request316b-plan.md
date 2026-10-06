# R316B: independent structural execution-day and exact-acceptance cohort pilot

Preregistered before event acquisition, market outcomes or labels. Parent:
R316 / draft PR88 at fa5612e49ce683ef661f5d6cdaee2a49889ab756.
Use R316B rather than a new numeric request to avoid the concurrent Work track.
Do not edit research-request.json, main, upstream models, or PR88.
June HOLD and July-August remain sealed. May is reused development, never HOLD.

## Population, clocks and staged freeze

1. Corporate-action census: every Massive reverse split executing May4-20,2026.
   No HOT admission or momentum exclusion. Preserve all raw census records.
   Execution day is primary; trading-day offsets -1,+1,+5 are descriptive.
   Only May1-29 sessions allowed. Membership is constructed from event data,
   never subsequent prices; save census before any market requests.
2. Catalyst pilot: market-wide classified 8-K filing-date May11-20 census,
   deduplicated by accession/ticker. Select an accession iff the first byte of
   SHA256('R316B|'+accession) is below16, independently of categories/outcomes.
   No replacement or support-dependent expansion. Store categories as multiple
   labels; an 8-K need not be good news or a company's first public announcement.
   Resolve CIK from the source or historical ticker reference, then retrieve the
   raw SEC filing header. Require matching ACCESSION NUMBER and FORM TYPE 8-K
   and raw ACCEPTANCE-DATETIME. Parse the timezone-free header as America/New_York
   with DST. Never assume a filing date is a precise intraday time or blindly
   trust a submissions JSON 'Z' suffix. Missing/blocked headers remain unknown.
3. Acceptance is not publication. Fixed earliest research eligibility is
   acceptance+180s (a declared dissemination assumption, not observed receipt),
   rounded up to a second. Use max(eligibility,09:35 ET); roll after-hours filings
   to next session09:35. Record timing bucket and original acceptance.
4. Split clocks are09:35 ET on each declared session. Corporate-action history
   establishes retrospective structural membership, not PIT foreknowledge.
   Query prior30-day ticker news and preserve reverse-split-title candidates and
   timestamps as an announcement audit; do not silently declare them linked to
   this exact execution or use a future execution to claim causal entry at -1.
   Split policy promotion is forbidden until announcement linkage is audited.

## Matching before target paths

Historical common U.S. stock (reference type CS, locale us, market stocks), prior
session nominal close in[0.50,20] and positive close*volume. On execution day,
use prior close*(split_from/split_to) to match prices on the new share basis;
the unadjusted mechanical split price jump is never counted as a return.
Retain all eligibility failures in a ledger; no outcome replacement.

For each eligible event anchor match three different common stocks from that
same prior session: squared log-price plus squared log-dollar-volume distance,
each coordinate within a factor4, deterministic ticker tie break. Exclude the
event ticker and tickers in the acquired split/8-K census within preceding
30 calendar days through that session. Historical reference is checked before
selection. Controls may be reused across sets and do not imply no real catalyst
(news/social/unknown 8-K coverage is incomplete). Fixed matching records and
identity hash are saved BEFORE any target-day second history is requested.
Fewer than3 controls remains explicit, never filled with later/price winners.

## Diagnostic outcomes

Fetch unadjusted complete paginated regular-session one-second bars only for
the frozen event/control identities. Next open at/after submission is a modeled
reference, not a guaranteed executable NBBO fill. Fixed horizon60min/session cap,
BASE costs and -2.5% absorbing stop exactly as R306 entry_labels. Light/stress
are fixed sensitivity scenarios on the same BASE stop path. Report whole-path
and stopped-path net5/10/20 ceilings and fixed-horizon/stop terminal return.
Ceilings are future-knowledge opportunity diagnostics, not earned returns.
No model fitting, policy tuning, threshold/stop/cost changes or MoE/router.

Never mark missing histories or missing terminal fills negative. Preserve next
print pending delays, largest interprint gap, >=60s gap flags and stop fill gap;
gaps are not identified halts, and quote size/market impact/official halt
coverage is absent. Completeness requires the terminal next print before close;
session-close caps without a regular next print remain censored.

Report eligibility/matching/acquisition/label attrition, complete-case prevalence
and all-member bounds (unknown outcomes all0 vs all1). Primary comparisons:
execution-day split and exact-acceptance 8-K separately, equal anchor weighting,
event minus mean controls, complete matched sets only. Fixed1000 paired bootstrap
draws by event ticker (preserve reused controls) and trading day; interval is
descriptive because matched observational sampling cannot prove causality.
Also list shared control identities and declare the bootstrap does not model
all dependence induced by shared controls.

The pilot can justify expanded preregistered development, never a specialist or
deployment. Adequate support requires40 complete pairs,6 event positives on
3 dates, >=80% complete pairs, and >=80% exact header resolution for the
selected catalyst census. A promising distribution additionally requires net5
rate ratio>=2 and paired-difference95% intervals above0 under both cluster
schemes. Otherwise report underpowered, timing-blocked, incomplete, or
not-supported. Do not expand dates/sampling in response to a result.

## Execution and evidence

One branch-only push workflow with explicit [run-r316b] message. Separate
artifacts and source SHA; never the main request dispatcher. Preserve event
census, timing ledger, frozen matched manifest, outcomes, raw provider cache,
SEC headers, request counters, input/content hashes and plan in artifacts.
After the official run remove the one-shot workflow and retain the findings
in a draft independent PR; no merge into main during this task.

Sources: Massive current /stocks/v1/splits documentation (execution is overnight,
including premarket), SEC EDGAR API documentation and webmaster FAQ
(acceptance versus public availability). Exact acceptance timestamps are not
asserted to be exact first-public catalyst timestamps.

## Local source-feasibility stage (before acquisition, no price outcomes)

Public publication of the prepared branch is blocked pending explicit user
approval by automatic review. Independently verify SEC raw-header availability
on the full official daily-index 8-K census May11,12,13,14,15,18,19,20. Apply the
same fixed accession hash sample (first byte<16), deduplicate accessions, keep
CIK from the official index, and save the complete selected membership BEFORE
retrieving headers. Preserve raw daily indexes and validated headers.
This SEC-only sample is a source feasibility check, NOT the Massive classified
catalyst cohort, NOT a price test, and never substitutes for its frozen membership.
Report coverage, time buckets, acceptance-to-filing-date discrepancies and
same-day clocks; do not increase sampling/date range to repair support.
