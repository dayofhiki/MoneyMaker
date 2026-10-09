# R333 complete: aggregates accessible; historical trades and quotes denied

One official qualification finished and its archived-source replay matched
exactly. The existing REST account accepted seconds/minutes for all12 fixed
sample windows. Its first trade request and first quote request returned HTTP403
with provider status NOT_AUTHORIZED and an explicit entitlement message. Under
the registered stop rule, each stream's remaining11 windows was unrequested.
This is an observed access barrier, not evidence that market records are absent.

| Source | Declared windows | Complete observed windows | Returned rows | Denied requests | Skipped after denial |
|---|---:|---:|---:|---:|---:|
| Seconds |12|12|81|0|0|
| Minutes |12|12|12|0|0|
| Trades |12|0|0|1|11|
| NBBO quotes |12|0|0|1|11|

Exactly26 HTTP attempts,11,921 retained response bytes; no retries, partial-page
limit, credential withholding or purchase. All26 wire bodies and sanitized
payloads were retained and authenticated. Every declared pair remains in the
report. The full1235-identity timestamp-window manifest remains authenticated,
but the full census was not collected. Selection was timestamp/identity-only:
one lexicographically first identity per original day, first60s of each proposed
window. These transport samples cannot estimate full account or target support.

R331's primary resolution34.8352% and90% failure remain unchanged. No gap cause
was inferred from the81 returned second bars. No conditions/corrections were
filtered, no economic labels or value fits were created, no policy promoted and
no sealed dates opened. The immediate bottleneck for independent reconciliation
is access to trade/quote observations, beyond the earlier source-provenance gap.

## Authentication and reproduction

Preregistration `e8e85dd4cb5410f5948b1d062720c963a87c02ec`; evaluated source
`5471e097cda5ae310ab09fcd25507254aa1d2823`, tree
`fbb286c438671dbe7fec4e64393c859b6076c18a`.
Official[run37918134023](https://github.com/dayofhiki/MoneyMaker/actions/runs/37918134023)
succeeded once. Source-page artifact11610023036 was published10:37:48Z before
offline analysis10:37:50Z. Source ZIP SHA256
`309a3b2967db8707d2ca706f4bdd711e874c818f5adcb83326ecb14fb21bf97d`.
Complete artifact11610362865 was published10:37:51Z; ZIP SHA256
`f4475837b45eb3041c4fb29ef6cdd3841433abb22388b5524f9e7b9a631c697b`.
Canonical result SHA256
`48bf3c10d5012f2c4846cb6dff9898ff8c393bd47b23a80824628c1f243dff7d`.
Page manifest SHA256
`d7a74b6e3b23fde180e36468bf68b10efe483326c18c2cca8cacb31d82bf5411`.

Archived-page reproduction made zero market requests. All996 JSON numeric fields
have zero error and report bytes are identical; categories, missingness and all
48 pairs match.26 original wire-page digests and26 sanitized payload digests were
verified, along with the full identity-window digest.84 relevant tests passed,
including32 new synthetic cases; critical lint passed. Both full source CI runs
37918133952 and37918137157 passed; the retained log reports1,289 tests and160
existing warnings. No failed or duplicate official qualification was selected.

The complete report, page provenance and replay are committed under research.
The original archived response bundle is also preserved for subsequent work.
Draft[PR106](https://github.com/dayofhiki/MoneyMaker/pull/106) remains unmerged,
stacked on R332. The one-shot qualification workflow is removed at closeout.

## Next bounded step

Do not buy an upgrade or claim that REST rejection proves every account access
route is unavailable. The repository already has a separate configured Flat
Files transport whose aggregate access CI passes. R334 registers a read-only
HEAD check of documented daily trade/quote objects for the SAME12 dates. It
checks metadata only, without downloading files or reading market rows. Its
result can settle whether the existing alternative transport is accessible and
show potential file sizes before any acquisition budget is proposed. HEAD
success is not GET/record completeness or causal quote availability evidence.
Only after a separately registered actual-source collection/eligibility design
can source reconciliation and later continuous ENTER/WAIT learning proceed.
