# Request 170 — historical trade-tape feasibility result

Authoritative run: `36031439436`  
Artifact: `10822905754`

No new market dates were opened.

## Decision

**FAIL — unavailable under current Massive entitlement**

The first historical trade query returned HTTP 403. Therefore:
- sampled states: 50;
- queried states before stop: 0;
- prior-60s trade coverage: 0%;
- prior-60s >=5 trade coverage: 0%;
- final-10s trade coverage: 0%.

This is an authorization/data-access failure, not evidence that the underlying
market states lack trade activity.

## Research consequence

Retire historical tick-trade enrichment under the current data entitlement.
Do not weaken the lookback windows or coverage gates.

The next branch should use the still-accessible historical one-second aggregate
feed to construct genuinely new causal post-entry microstructure proxies:
directional volume/transaction imbalance, short-burst persistence, path
efficiency, sign-flip/chop, seconds since local extrema, close-location and
VWAP displacement.

Evaluate those features on the surviving multi-minute remaining-option target
before attempting another executable recurrent controller.
