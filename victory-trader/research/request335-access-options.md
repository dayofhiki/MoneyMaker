# R335 source choice after the completed R333/R334 access checks

Update 2026-10-09: the user has confirmed Stocks Starter ($29/month), with no
approval for paid upgrades, purchases or metered calls. The current comparison,
conditional zero-cost qualification route and exact integrity requirements are
in [request335-cost-integrity-review.md](request335-cost-integrity-review.md).
The earlier plan-unknown text below records the previous preparatory stage;
Starter does not include the required trade/NBBO tick streams. Advanced is a
price benchmark, not the selected next purchase.

The minimum next input is the existing account's stocks plan/entitlement or an
already licensed historical trade + NBBO export. No API keys should be shared
in a message. No account plan is inferred from the original 403 observations.

Public individual-use plan documentation checked 2026-10-09:

| Existing plan | Listed monthly USD price | Historical trades REST / Flat Files | Historical NBBO REST / Flat Files | Implication for this fixed census |
|---|---:|---|---|---|
| Stocks Starter |29|Not included / Not included|Not included / Not included|Does not list both required tick streams|
| Stocks Developer |79|Included, 10 years / Included, 10 years|Not included / Not included|Trades alone cannot supply the joint contract|
| Stocks Advanced |199|Included, all history / Included, all history|Included, all history / Included, all history|Listed datasets cover May 2026; actual configured account access remains unverified|

These are public advertised monthly prices, not the user's subscription, a
checkout quote, incremental upgrade charge or purchase authorization. Prices,
taxes, prorating and applicable license terms must be settled before a purchase.
Individual plans are advertised for individual use; no business service or
redistribution rights are assumed by this research preparation.

Primary sources:
- https://massive.com/stocks
- https://massive.com/docs/rest/stocks/trades-quotes/trades
- https://massive.com/docs/rest/stocks/trades-quotes/quotes
- https://massive.com/docs/flat-files/stocks/trades
- https://massive.com/docs/flat-files/stocks/quotes

## Concrete follow-on paths

1. If the existing account already includes both streams, verify the account
   and configured transport association without publishing credentials. Then
   separately register a small fixed-date check with a known positive control,
   error-code-only diagnostics and bounded requests. R334's 403 HEAD does not
   settle whether signing, method or entitlement caused the Flat Files rejection.
2. If already licensed exports exist, supply their declared schema, exact bounds,
   byte hashes and acquisition evidence. The implemented offline inventory checks
   explicit files and preserves all 1,235 identities / 2,470 required tick-stream
   entries. It still does not declare a page-complete or causal source from hashes
   alone. No actual trade/quote export was supplied to this preparatory run.
3. If neither route is available, a new subscription or alternative vendor is a
   separate spending decision. Do not make another denied request or claim a new
   model result as a substitute. The $199 Advanced listing is a concrete access
   option, not a recommendation that the research will justify its cost.

An existing license can avoid new spending. Full daily whole-market files have
no established size budget from R334: error Content-Length 65 is not a market
file size. Prefer a separately registered, explicitly bounded per-ticker export
or REST page-chain design once access is known; do not start a daily GET here.

## Offline input interface now available

Run `python -m victory_trader.offline_source_inventory` with `--previous`, `--pins`,
`--plan`, `--catalog`, `--export-root` and an unused `--output` directory. The
original pins/plan must be exactly the committed R333 files. `--previous` contains
the three authenticated R332 files under `official332/`. The catalog is a JSON
list; an empty list is explicitly supported and does not inspect the export root.

Each catalog entry supplies `trading_day`, `ticker`, `stream` (`trades`/`quotes`),
integer `start_ns`/`end_ns`, `timestamp_unit` (`nanosecond`), `format` (`rest_jsonl`
or `provider_csv`), relative `file`, original `sha256`, `source_ref`, and relative
`provenance_file`/`provenance_sha256`. Each partition is one ticker/day/stream;
mixed-ticker, whole-market, compressed or parquet files need a later adapter
with its own explicit byte contract. JSONL keeps original REST row objects; CSV
requires the original provider header and exact decimal integer SIP timestamps.

Schema/hash-valid exports are `SOURCE_COMPLETENESS_UNVERIFIED`. Partial bounds
remain `PARTIAL_DECLARED_WINDOW`; missing entries are `NOT_SUPPLIED`; invalid or
unavailable files remain explicit. No field or returned row is a fill, an eligible
trade, a strategy receipt time or an economic target in this stage.
