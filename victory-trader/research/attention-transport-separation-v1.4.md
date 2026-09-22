# Attention/transport separation v1.4 — research direction

## Decision

Do not force the predictive shortlist to satisfy a historical per-ticker REST
download proxy. Requests 129 and 130 both met the proxy while materially
damaging causal runner capture. April 2026 and later remain sealed.

## Evidence

- Rank-40 hysteresis in request 128 preserved capture within 0.92 percentage
  points, retained 96.14% of learned-focus captures, raised set retention to
  76.31%, and reduced slot replacements from 8.19 to 4.78 per minute.
- A hard 300-symbol ceiling reduced ticker-days by 46.35% but lost 13.39
  percentage points of capture on fresh March sessions.
- Linear release of the same ceiling still lost 8.97 percentage points on the
  development block.

The unique ticker-day count describes how the historical one-second REST cache
is populated. It is not itself a model-quality metric, and treating it as a
hard prediction constraint suppresses real late-session opportunity migration.

## Next implementation target

Keep prediction and transport concerns separate:

1. retain the learned focus-60 and rank-40 persistent shortlist-20 as the
   attention candidate;
2. maintain at most 20 concurrent high-resolution subscriptions;
3. measure actual subscription additions/removals, active duration, delayed or
   missing second bars, and reconnect behavior in the runtime;
4. cache historical second bars independently of model admission so REST
   request count cannot change the selected population;
5. only after transport simulation passes, preregister one integrated
   shortlist -> one-second -> HOT-10 evaluation on the sealed holdout.

## Non-negotiable gates

The next transport implementation must not change selected ticker membership
to meet download cost. It must preserve causal timestamps, never exceed 20
concurrent subscriptions, expose churn and data-freshness telemetry, and pass
fault-injection tests for missing bars and reconnects. No ENTRY/ABSTAIN work
resumes until that bridge is verified.

## Initial implementation

`ObservationTransport` now reconciles an externally selected set without
reranking or dropping names, enforces the concurrent capacity, records
subscription additions/removals, flags missing and stale bars, rejects
future-dated events, and requires explicit recovery after disconnect. Unit
tests inject capacity overflow, missing data, stale data and reconnects.

The bridge is now event-driven rather than tied to one-minute scheduling.
`RankHysteresisSelector` can reconcile whenever market scores change, and
`ObservationBridge` passes its exact selection to transport without reranking.
The integrated hierarchy now uses this rank-40 bridge for its 20-name
high-resolution candidate set.
