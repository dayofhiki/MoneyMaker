# R334: bounded metadata-only check of the existing Flat Files route

Register before execution, following exactly replayed R333. REST trade/quote
requests were denied; existing aggregate Flat Files CI succeeded. Do not infer
the alternate route's trade/quote entitlement from either result. No purchase,
account change, credential search or raw market download is part of R334.

Authenticate `research/results/request333.json` and `request333-plan.json` by
`request334-inputs.json`. Use their exact12 original May dates. Query the documented
`us_stocks_sip/trades_v1/YYYY/MM/YYYY-MM-DD.csv.gz` and
`us_stocks_sip/quotes_v1/YYYY/MM/YYYY-MM-DD.csv.gz` keys in bucket flatfiles, HTTPS
files.massive.com.24 pairs, sorted dates then trades/quotes. No other prefixes,
listing, day, account or market rows. This retains the prior full identity plan;
daily-file metadata applies to whole files and does not establish ticker coverage.

Use only the repository's explicitly configured MASSIVE_S3_ACCESS_KEY and
MASSIVE_S3_SECRET_KEY. Missing either: all pairs remain unrequested. Sign HEAD
with S3v4, service s3, region us-east-1, path addressing. Direct requests.head,
stream=True, allow_redirects=False, timeout10s. At most24 calls, at least0.5s
between starts, zero retries, redirects, GETs, body reads or file downloads.
No SDK automatic retries/region redirects or default credential discovery.

Record each key, request/response UTC, HTTP status, numeric Content-Length when
valid, ETag/Last-Modified where present and exception TYPE only. Never store
authorization headers, credential values or raw error text. A redirect remains
unresolved; no follow-up URL is tried.403 means this HEAD was rejected;404 means
not-found-or-undisclosed, not proof of no records. Successful HEAD only establishes
observed metadata access, not GET permission, complete/correct bytes, per-ticker
records, historical receipt times or a fill proxy. Keep every failed/missing pair.

Publish the metadata ledger before summarization. Replay the immutable ledger
locally without network, comparing all JSON fields including timestamps/statuses
and numeric file sizes. Synthetic tests cover success/missing headers, denial,
missing credentials, exceptions, fixed scope, signing secrecy, request spacing,
and immutable records. One official run; preserve failures, no silent rerun, and
retire the one-shot workflow at closeout.

No market data rows, economic labels, fits, policy promotion, new dates or costs/
stops are changed. R331's90% failure remains. If all metadata checks succeed,
record the file-size envelope and prepare a separate budgeted GET/normalization/
eligibility design before collection. If they fail, report actual dataset access
as the block and prepare an explicit user-reviewable access choice. Do not select
an order contract or infer returns from either outcome.

Documented prefixes checked2026-10-09:
- https://massive.com/docs/flat-files/stocks/trades
- https://massive.com/docs/flat-files/stocks/quotes
