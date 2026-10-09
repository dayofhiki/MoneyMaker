# R334 complete: all24 fixed trade/quote metadata HEADs rejected

The existing configured S3 account returned HTTP403 for each of the12 trade
objects and12 quote objects on the original May dates. One preregistered run
completed, and its immutable ledger was exactly reproduced without network.

| Stream | Declared daily objects | HTTP attempts | HEAD200 | HEAD403 |
|---|---:|---:|---:|---:|
| Trades |12|12|0|12|
| NBBO quotes |12|12|0|12|

The dates were May5,6,7,8,11,12,13,14,15,18,19,20; no date or object was selected
after observing access. Credentials were configured. Direct signed HEADs used
the registered endpoint, prefixes and keys with no redirects, retries, GETs,
listing, purchase or account modification. No market body bytes or rows were
read. The observed minimum request-start spacing was0.500097s.

This establishes observed HEAD rejection on these keys, not absence of market
records or a definitive Flat Files denial cause. Without an error body or a
same-run positive HEAD control, R334 cannot distinguish entitlement, signing,
method or gateway causes. The preserved Content-Length65 is an error-response
header, not the compressed market file size. There is no usable file-size
envelope. Zero known accessible file bytes must not be reported as zero-size
files. R333 independently retained an explicit REST entitlement denial for its
first trade and quote requests; these are separate transport observations.

## Authentication and replay

R334 was preregistered in R333 closeout commit
`fc5abd2daa1f689c98a9d17119195420042b2222`, before implementation/execution.
Evaluated source `f373f21c588d1cca6f3285a1c486f9eb913a10e1`, tree
`daa020b4f4e4d5493c99835e215c3e01987b1cf6`.
Official[run37920390526](https://github.com/dayofhiki/MoneyMaker/actions/runs/37920390526)
succeeded once. Metadata artifact11611851108 was published10:54:37Z, before
offline analysis started10:54:37.660275Z. Metadata ZIP SHA256
`9b899f1f875c650f8e560c55406889e7fb6fec6a49bf38995d7cc424a3627ff4`.
Complete artifact11611452568 was published10:54:40Z; ZIP SHA256
`14c11e5d1a59dcf184ffea86acff16c5d17680c4d761fae770741d80a270b19d`.
Canonical result SHA256
`c74ec3b75cd1e94a54078980b8aa5450dd7abf3c2984834b67069e597795f97b`.
Metadata ledger SHA256
`68bd378f8e89091bf96581ac0afeba3d0674ccf5b7e977c01d8962524d1dd008`.

The authenticated archive contains the evaluated source SHA and the exact
metadata ledger. Local replay makes zero HEAD/GET requests. All68 numeric
fields have zero error; canonical report bytes, all24 keys, statuses, UTC
timestamps, categories and unknowns match exactly.44 transport/scope/secrecy
tests passed locally and officially, including12 new cases. The local full
suite passed1,301 tests with160 existing warnings; critical lint passed.
Both full source CI runs37920390530 and37920420600 succeeded; the retained
source CI log confirms1,301 tests and160 warnings. R333 closeout CI37919707004
also succeeded before this stage.

The complete canonical result, metadata ledger, provenance and replay are
committed. Draft[PR107](https://github.com/dayofhiki/MoneyMaker/pull/107) remains
unmerged and stacked on R333. The one-shot R334 workflow is retired at closeout.
No failed or duplicate official run was substituted for this run.

## Economic implication and next action

Required independent trade/quote observations remain unavailable through the
two tested routes. The source-gap cause remains unidentified. R331's34.8352%
primary resolution and90% gate failure remain unchanged. This is no evidence
of higher returns, a tradable order, or learned ENTER/WAIT support. No economic
labels, model fits, policy promotion, live orders or sealed dates were added.

`request335-proposal.md` specifies a reviewable minimum data/access choice.
It is a proposal, not permission to buy access, search credentials or repeat
the same rejected calls. Once an authorized observation source exists, register
its acquisition/normalization/causal-eligibility contract before collection,
then reconcile the fixed census before any economic-target or action-learning
intervention. Continuous decisions and after-cost account outcomes remain the
research objective; additional model fits cannot erase the missing observations.
