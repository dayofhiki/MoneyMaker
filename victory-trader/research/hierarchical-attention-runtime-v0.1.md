# Hierarchical attention runtime v0.1

## Status

Implemented as the first executable architecture milestone under `ROADMAP.md`.

This milestone changes the shape of the system, not the claimed profitability
of any research model. It introduces a causal runtime boundary into which the
existing opportunity, continuation, remaining-option, and recurrent-exit
models can be connected without treating any fixed horizon as the final
trader.

No sealed month was opened and no development threshold was tuned from market
outcomes for this work.

## Runtime contract

One `AttentionRuntime.step(timestamp, market_batch)` call represents a
completed low-cost broad-market scan at one timestamp. The runtime maintains
persistent per-ticker state and emits the observation plan for the next cycle.

States:

```text
SCAN -> WATCH -> HOT -> POSITION -> WATCH / DROP
```

Observation tiers:

| State | Requested observation tier | Purpose |
| --- | --- | --- |
| SCAN | grouped minute | retain broad eligible-market coverage |
| WATCH | per-symbol minute bars | build richer path state |
| HOT | second bars + NBBO | support entry/abstention decisions |
| POSITION | trades + NBBO | support recurrent HOLD/EXIT and execution |
| DROP | none | release per-symbol resources; broad scan may rediscover it |

The provider adapters do not exist yet. The emitted feed names are explicit
requests for the acquisition layer, not a claim that these feeds are already
being collected live.

## Frozen v0.1 mechanics

- Every observed ticker remains represented; low-ranked names return to SCAN
  rather than disappearing from the market view.
- WATCH and HOT have separate finite capacities.
- Candidates compete by descending point-in-time attention score with ticker
  as a deterministic tie-break.
- A stronger new candidate can displace a cooling candidate immediately.
- Separate enter and exit thresholds provide hysteresis against state chatter.
- Open positions never compete for WATCH/HOT capacity.
- A just-closed position receives one WATCH cycle so exit/re-entry reasoning
  remains continuous.
- Invalid or ineligible symbols enter DROP but may be rediscovered by a later
  valid broad scan.
- Missing non-position symbols age out after a configurable number of broad
  scan batches.
- Missing position observations remain pinned at POSITION and surface
  `position_feed_missing`; the future external safety layer must react to this
  condition.
- Batch timestamps must strictly increase and duplicate tickers are rejected.
- Runtime state has a JSON-serializable checkpoint and restore path.

## Separation from model research

`attention_score` is deliberately an input, not a formula hidden in the
runtime. The score must eventually be supplied by a causal attention model
trained and evaluated under the research protocol. This prevents the
orchestrator from silently becoming another hand-tuned trading strategy.

Likewise, HOT is not BUY. It means only that the ticker deserves expensive
observation. Entry/abstention remains a separate economic decision. POSITION
does not encode a fixed holding period; it reserves the highest observation
tier for recurrent HOLD/EXIT inference.

## Tests

The v0.1 suite covers:

- broad-market SCAN retention;
- ranked WATCH/HOT promotion;
- attention displacement by a newly emerging leader;
- hysteresis;
- pinned position monitoring and post-exit WATCH;
- invalid-data DROP and rediscovery;
- missing-observation aging;
- causal timestamp and duplicate-batch guards;
- checkpoint/restore continuity.

## Next registered implementation bridge

The next milestone should be an offline event-replay adapter, not a new
profitability claim.

1. Convert historical grouped-minute/minute state panels into chronological
   market-wide scan batches.
2. Define a point-in-time attention-score interface and first non-optimized
   baseline using already available causal fields.
3. Record state transitions, tier occupancy, promotion lead time, missed-runner
   rate, and estimated data/compute demand.
4. Replay the exact same days under multiple *capacity budgets*, without using
   trade returns to choose thresholds.
5. Only after coverage behavior is audited, connect HOT states to the frozen
   entry/abstention policy and POSITION states to the recurrent HOLD/EXIT
   controller.

This ordering tests whether dynamic attention can preserve opportunity
coverage before allowing downstream P&L to tune the scanner.
