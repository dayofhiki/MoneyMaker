# Request 170 — historical trade-tape feasibility

## Purpose

Requests 166-169 show that the current minute/second/path state does not
generalize for executable HOLD/EXIT timing. Request 169's exact 2/5/10-minute
value models all reversed sign on June 8-12.

Historical NBBO is unavailable under the configured entitlement. Before
building another model, Request 170 tests whether historical tick-level trades
are accessible and dense enough to provide genuinely new causal intraday
information.

## Data

No new dates.

Deterministically sample 10 Request-166 evaluation POSITION states per day from
June 8-12, 50 states total. Sampling is sorted by day/time/ticker and evenly
spaced. No outcome participates.

## Causality

For each sampled state at t query only historical trades with SIP timestamp in:

    [t - 60 seconds, t]

No future trade may enter the probe.

The feasibility report may calculate:
- 60s and 10s trade counts/volume;
- last-trade age;
- first-to-last tape return;
- tick-rule signed-volume imbalance;
- median intertrade time.

These are only data-availability diagnostics.

## Frozen feasibility gate

Proceed to a tape-enriched POSITION model only if:

1. no historical-trade 401/403 authorization failure;
2. every sampled state is queried;
3. >=80% of sampled states have at least one prior-60s trade;
4. >=70% have at least five prior-60s trades;
5. >=60% have a trade in the final 10 seconds.

Do not loosen windows or coverage thresholds after inspection.

If unavailable, retire tick-tape enrichment under current entitlement and move
to a different information/target branch.
