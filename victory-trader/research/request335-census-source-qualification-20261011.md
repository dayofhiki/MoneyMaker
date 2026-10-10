# R335 — source requirements retained during original-census acquisition

This is a documentation and acquisition-diagnostics review, with no execution
labels, market requests or spending. The previous source gate remains false.
The complete original denominator and90% execution-resolution requirement remain
unchanged. A terminal API marker is not qualification of complete market history.

The official CLI schema was retrieved again at the already registered upstream
commit `53606273aa230a40c64b783425dcb3f4423ede30`. Its476,591 bytes reproduce SHA256
`56b0278e0eca3dde122304b5dd6b74abfab6848123141a7947bc15614d8843b7` exactly.
This verifies the prior schema reference rather than relying on its old summary.
The stock historical endpoints reference the corresponding trade/quote schemas.

| Requirement | Current evidence and remaining work |
|---|---|
| Exact timestamps and bounds | Raw RFC3339 values, integer nanoseconds and original half-open ranges are retained. The schema specifies representation/precision; it does not identify historical SIP, participant or strategy receipt time. |
| Historical quote size unit | The pinned historical schema and2025-10-30 change notice specify shares from2025-11-03. May2026 quantities are retained unchanged. Older real-time documentation still shows round lots; its examples do not supersede the date-specific historical schema. |
| Historical trade size unit | The pinned historical trade schema describes size without a physical unit. The adapter keeps its unit unverified. No unit is inferred from observed magnitudes or a different feed. |
| Equal-clock order | Historical responses are sorted by symbol and provider timestamp. No inspected contract specifies a causal total order among equal timestamps or across trades/quotes. No file order or exchange-supplied trade ID becomes an execution sequence. |
| Corrections/cancels | Historical optional update status describes the current API record. Real-time corrections/cancels are separate messages. A current record or absent status does not supply original receipt, correction delivery or strategy-visible history. |
| Quote conditions and halts | The historical schema defines one condition for both sides or bid/ask order for two flags, and zero price as inactive. Eligibility, date-specific rule application, halt/LULD/status coverage and stale state remain separate requirements. No invalid/locked/crossed/inactive state is silently filtered into an executable quote. |
| Universe and market completeness | SIP and OTC are separate feed choices. API terminal markers cover only a query chain; historical listing/identity/venue/OTC/inactive-symbol completeness must still be evidenced over all original identities. Empty rows remain observations of this query, not proof of no market activity. |
| Availability/version | Reused and newly downloaded pages have distinct acquisition provenance. asof- disables symbol remapping; it does not freeze provider revision history. Download/HTTP timestamps are acquisition metadata, not historical strategy receipt. |
| Actual execution and account economics | NBBO top-of-book quantities do not prove routing, depth, queue priority, fills or slippage. Registered causal availability/latency/quote-age/participation/cancel/fees/capital/opportunity-cost rules and outcome resolution are required before economic evaluation. |

Full-census row diagnostics will measure these issues only where fields can be
observed. Missing fields/clock semantics remain unknown, rather than reconstructed
with interpolation, convenient latency assumptions or profitable outcomes. No
source-validation percentage is substituted for execution-resolution percentage.

Original quantities, equal-clock rows and any duplicate observations remain in
private raw evidence. Aggregated row counts are original-partition observations;
overlapping original windows may count the same market event more than once.
Cross-stream equal-clock counts are paired by exact original identity and window,
not merely by symbol/date. This avoids overwriting different windows for one ticker.

## Primary references reviewed

- [Pinned Alpaca CLI historical OpenAPI](https://github.com/alpacahq/cli/blob/53606273aa230a40c64b783425dcb3f4423ede30/api/specs/market-data-api.json): historical schemas, physical quote-unit date rule, optional trade update and timestamp representation.
- [Quote-size change notice,2025-10-30](https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change): effective2025-11-03.
- [Historical trades](https://docs.alpaca.markets/us/reference/stocktrades-1) and [historical quotes](https://docs.alpaca.markets/us/reference/stockquotes-1): timestamp sorting, inclusive REST boundaries, page tokens, SIP/OTC separation and asof symbol-mapping semantics.
- [Real-time stock data](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data): separate correction/cancel messages and status/LULD channels. This is not evidence that those histories are present in the downloaded REST response.

No paid plan, support message or alternative data acquisition is initiated by
this review. June HOLD /July·August SEALED; source_ready=false; qualified0;
economic labels0; profitability incomplete.
