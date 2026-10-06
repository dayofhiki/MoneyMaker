# R316B: first structural pilot is underpowered; exact-header transport recovery is prepared

Independent continuation of R316 / draft PR88. Explicit publication approval
was received2026-10-06, and draft PR89 is stacked on PR88. Main and the Work
research request remain unchanged. First official run37458825600 completed
from c07bd899d3872f389222f7a3917be0a7ac2d3f7b; artifact digest:
`sha256:a6778e07ee100eea153b035f881c3ccb119a5b5fbdae3cde368064c822106236`.

The initial structural census includes55 execution events and220 pre/post
anchors. Historical common-stock/price/reference eligibility yields69 matched
anchors and276 event/control rows. Primary execution-day support is20 pairs:
BASE-stopped net5 opportunity4/20 vs2/60 controls (20% vs3.33%, point ratio6).
Both clustered95% difference intervals cross0, and the fixed support floor
fails. This is **underpowered**, not promoted. Whole-path net5 opportunity10/20
is a different, future-knowledge quantity; the simple60min/stop terminal mean
is -1.73%, so opportunity prevalence is not a profitable policy.

The runner's SEC access denial left all115 ticker/accession pairs timing-unknown;
no negative catalyst conclusion is made. All90 unique accessions are already
covered by the200 verified **pre-price** SEC headers described below. Recovery
uses that complete fixed source seed and the exact original provider census,
without new dates, altered sampling, labels or parameters.

Fourteen artifact hashes were verified;276 original outcome rows and all5
reports reproduce exactly from275 cached histories with **zero new network
requests**. Evidence: request316b-first-reproduction.json. The recovery workflow
also requires all original structural rows and four reports to remain exact.

## Completed empirical source-feasibility experiment

Before any price acquisition, freeze the complete official SEC daily-index
8-K census for May11,12,13,14,15,18,19,20,2026 and select accessions by the
preregistered SHA256 first-byte<16 rule. Membership was written before any
selected raw header was retrieved. The sample does not depend on price,
company performance, category, model score or a winner list.

| Diagnostic | Result |
|---|---:|
| Official daily-index 8-K census | 2,822 |
| Fixed selected accessions | 200 |
| Validated raw acceptance headers | 200 |
| Exact header coverage | 100% |
| Premarket acceptances | 60 |
| Regular-session acceptances | 25 |
| After-hours acceptances | 115 |
| Same acceptance/filing date | 192 |
| Filing date one day after acceptance | 8 |
| Failed resolved sample members | 0 |

Raw `ACCEPTANCE-DATETIME` was parsed in America/New_York, including DST,
and every header had a matching accession and 8-K form. An index date is
therefore insufficient for intraday knowledge. The eight one-day discrepancies
are a concrete additional reason to reconstruct the clock rather than attach
an event using a date alone.

The completed acquisition used184 new header requests and16 already cached
validated headers from the interrupted sequential acquisition. The cohort and
sampling remained fixed when the latency transport was changed to eight
workers with serialized request starts no faster than five/second.

The fixed sample identity hash is
`0b47f5af8f3fe58d75e7b4ca1bd415ec206c6431c2a3fad44a388a30ae8863fe`.
All eight indexes, selected membership, timing ledger and200 raw headers are
preserved in the evidence bundle. No market prices, outcomes, June HOLD or
July-August were accessed during this empirical timing check.

## Interpretation and support limits

This establishes that SEC raw acceptance timing is practically recoverable
for this fixed source sample. It addresses R316's filing-date precision gap.
It does **not** establish a profitable catalyst distribution or a usable
low-price trading population. Only25 of these sampled filings were accepted
in the regular session; common-stock and historical price eligibility have
not been established yet. The SEC-only feasibility census is distinct from
the pending Massive classified/ticker-linked catalyst cohort.

Acceptance is not necessarily the first public announcement and does not
prove immediate public dissemination. Acceptance+180s is a preregistered
research availability assumption, not an observed first-public timestamp.
Premarket eligibility is09:35 ET, and after-hours acceptance rolls to the
next regular session09:35. No specialist or router is promoted.

## Prepared independent next experiment

The implementation and branch-only one-shot workflow are complete; the first
structural run above has finished, and exact-timing source recovery is next:

- Every reverse split executing May4-20, without the HOT/same-day-split
  exclusion; execution day primary, trading offsets-1,+1,+5 descriptive.
- Raw event census and announcement candidates before target price requests;
  announcement-to-execution linkage remains explicitly unverified.
- Exact-header 8-K anchors sampled independently of outcomes, with a timing
  ledger and no invented same-day knowledge after a source failure.
- Historical common-stock eligibility and three controls matched by prior-day
  price/dollar volume. Execution-day price matching uses the new share basis;
  mechanical split repricing is not a return.
- Frozen matching hash before any target one-second bars, next-open fill
  references, unchanged BASE costs, -2.5% absorbing stop and60min/session cap.
- Complete-case versus all-member censoring bounds, net5/10/20 opportunity
  ceilings, terminal returns, fixed light/stress sensitivities and pending
  print gaps. No official halt/quote/market-impact claim.
- Separate event-ticker and trading-day paired bootstrap diagnostics, with
  shared-control dependence disclosed and no policy/model refit.

Future-knowledge opportunity ceilings are not realized trading profits.
A reverse-split execution-day label is retrospective structural membership,
not proof that a causal entry policy knew the announced split in advance.
The pre-day offset cannot be promoted until announcement linkage is verified.

Validation: Ruff critical checks pass; full local suite **1,049 tests pass**.
The focused source/clock/membership tests also pass. No main workflow trigger,
research-request.json or upstream score/model was modified. The first official
GitHub cohort-price run and both push/PR CI runs passed. Recovery validation
adds seed integrity checks and preserves the original structural results.

## Approved next action

Run the exact-header source recovery on the approved independent branch and
draft PR89 in `dayofhiki/MoneyMaker`. Inspect/fix acquisition and integrity
failures, record results and artifact hashes, remove the one-shot workflow
before any future merge. Keep main/Work independent. Do not merge or introduce
an event-conditioned trading model from this pilot.

Official source references:
- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/files/about/webmaster-faq.htm
- https://massive.com/docs/rest/stocks/corporate-actions/splits
