# MoneyMaker Research Roadmap

## North Star

MoneyMaker is not intended to become a fixed-horizon predictor or a collection of isolated backtest rules.

The long-term target is a **stateful autonomous trader** that remains active while the market is open, continuously observes the market, dynamically allocates attention, chooses whether and when to enter, manages open positions as conditions evolve, and decides when to exit.

The system should ultimately behave more like a skilled active surge-stock trader than a one-shot classifier.

## Target Operating Loop

```text
MARKET
  -> LOW-COST MARKET-WIDE SCAN
  -> ATTENTION / WATCHLIST
  -> HIGH-RESOLUTION MONITORING
  -> ENTRY / ABSTAIN
  -> POSITION MANAGEMENT
  -> EXIT
  -> WATCH AGAIN OR DROP
  -> continue scanning
```

Capital allocation, risk controls, execution modeling, data quality checks, and persistent state surround this loop.

## Design Principles

### 1. Keep eyes on the whole market

The system must maintain a low-cost view of the broad eligible market rather than operating only on a static preselected ticker list. The purpose of the broad scanner is to detect emerging opportunities early enough for deeper inspection.

### 2. Allocate attention dynamically

Ticker attention is a changing state, not a permanent watchlist.

A ticker may move through states such as:

```text
SCAN -> WATCH -> HOT -> POSITION -> WATCH / DROP
```

New tickers must be able to enter the attention set at any time. Cooling tickers must lose attention so compute and capital can move elsewhere.

### 3. Use adaptive observation resolution

Do not process every ticker with the most expensive data and model.

Use inexpensive, lower-resolution monitoring for the broad market, then increase temporal and market-microstructure resolution as a ticker becomes more relevant. Candidate resolutions include minute bars, multi-second bars, 1-second bars, trades, and NBBO quote events.

The appropriate resolution should eventually depend on market state rather than being globally fixed.

### 4. Treat abstention as an action

A detected surge candidate is not automatically a trade.

The entry layer must learn whether committing capital has positive economic value given only information available at that time. Doing nothing is a valid and important decision.

### 5. Do not impose a fixed holding horizon

Five-minute, ten-minute, fifteen-minute, or thirty-minute horizons may be useful research instruments, but they are not the final policy.

The deployed trader should repeatedly observe the market and determine its own effective holding duration. A trade may deserve an exit after seconds or continued holding for much longer if the state supports it.

### 6. Continue reasoning after entry

Entry does not end inference.

While a position is open, new price, volume, trade, quote, liquidity, and state information should continuously update the decision process. HOLD and EXIT are recurrent decisions. Re-entry may later become part of the same state machine.

### 7. Optimize economic trading outcomes, not isolated predictive metrics

AUC, correlation, classification accuracy, and one-step value estimates are diagnostics, not the final objective.

The final evaluation must execute decisions sequentially and report economic outcomes including transaction/execution assumptions, trade distribution, drawdown, and synthetic account evolution.

Research summaries should include the audit account convention and report:

```text
starting balance -> ending balance -> total return
```

These are backtest/audit results, never live-return forecasts.

### 8. Make execution increasingly realistic

Current LIGHT / BASE / STRESS execution costs are transparent scenarios, not claims about actual fills.

Research should progressively replace coarse execution assumptions with empirical historical bid/ask information where available, including NBBO spread, quote freshness, and top-of-book depth for the intended order size. Market impact that cannot be identified from the available data must remain an explicit uncertainty.

### 9. Enforce strict causality

Every decision must use only information that would have been available at that timestamp.

Training, calibration, model selection, and evaluation must respect chronology. Do not leak future information through labels, preprocessing, normalization, quote selection, or state construction.

Do not repeatedly tune thresholds or hyperparameters against failed evaluation months until they pass.

Sealed periods remain sealed until a preregistered research bridge justifies opening them.

### 10. Expect edge decay

No edge is assumed permanent.

A mature system should monitor whether live or forward performance distributions depart materially from validated historical behavior. When confidence deteriorates, the system should be capable of reducing exposure or abstaining while the research pipeline investigates the change.

### 11. Allocate scarce capital across simultaneous opportunities

Eventually, BUY/NO-BUY decisions are insufficient.

When multiple HOT candidates exist simultaneously, the trader must decide which opportunities deserve attention and capital, how much capital to allocate, and whether an existing position makes a new opportunity unattractive under portfolio-level risk constraints.

### 12. Separate the trading policy from hard safety controls

The deployment path is:

```text
backtest -> untouched forward validation -> paper trading -> constrained live trading
```

Live execution must have external controls the model cannot override, including position limits, daily loss limits, stale/bad-data handling, order-state reconciliation, and a kill switch.

## Research Interpretation

Every experiment should answer:

1. **Which component of the final trader does this experiment improve?**
2. **Is the experiment causal and chronologically honest?**
3. **Does the result improve executable economic performance rather than only a proxy metric?**
4. **What uncertainty remains before this component can be promoted?**

A failed experiment is useful when it eliminates a hypothesis without contaminating sealed validation data.

## Current Architectural Interpretation

The work performed so far on opportunity detection, continuation, remaining option value, and recurrent exit decisions should be treated as components of the larger trader, not as the complete strategy.

In particular, fixed minute-level experiments are stepping stones. They must not silently redefine the final objective into a fixed-horizon strategy.

A likely mature architecture is hierarchical:

- **Market scanner:** broad, cheap, continuous coverage.
- **Attention manager:** dynamically promotes and demotes tickers.
- **High-resolution observer:** seconds/ticks/NBBO for WATCH/HOT names.
- **Entry/abstention policy:** commits capital only when justified.
- **Position manager:** recurrent adaptive HOLD/EXIT decisions.
- **Execution layer:** models and eventually submits realistic orders.
- **Capital/risk layer:** coordinates simultaneous opportunities and hard limits.
- **Monitoring/research layer:** detects degradation and supports edge renewal.

## Implementation Milestones

The roadmap is now being implemented as explicit, testable layers.

### Milestone 1 — stateful attention runtime (implemented)

`victory_trader.attention_runtime` provides the first executable orchestration
boundary:

- persistent `SCAN / WATCH / HOT / POSITION / DROP` ticker state;
- bounded WATCH/HOT capacity with score-ranked reallocation;
- hysteresis, missing-data aging, invalid-data DROP and rediscovery;
- position pinning and one-cycle post-exit WATCH;
- state-dependent grouped-minute, minute, second, and trades/NBBO feed plans;
- monotonic-time guards and JSON-serializable checkpoint/restore.

This milestone does not claim a validated attention score or a profitable
policy. It separates orchestration from prediction so later research models
cannot silently redefine the final architecture. See
`hierarchical-attention-runtime-v0.1.md`.

### Milestone 2 — chronological market replay (in progress)

Build an offline adapter that replays historical market-wide batches through
the runtime and measures coverage, promotion timing, tier occupancy, churn,
and data/compute demand. Capacity studies must not use evaluation returns to
tune score thresholds.

Phase 2A is implemented in `victory_trader.attention_replay`. It builds a
causal cross-sectional scan baseline, replays each session in timestamp order,
and audits runner capture plus observation-tier demand. Its input allowlist
ignores outcome columns. See `chronological-market-replay-v0.1.md`.

Phase 2B is implemented in `victory_trader.attention_flatfile_replay` and the
`attention_market_replay_v01` Actions operation. Request 107 completed the
frozen 2026-01-02 market-wide smoke after hardening per-day caches, duplicate
resolution, workflow failure propagation and missing-data capacity accounting.
The audited outputs have no duplicate keys and respect the fixed WATCH 50/HOT
10 budgets. Its coverage figures are not a threshold-tuning or profitability
test.

Phase 2C request 108 completed the unchanged replay through 2026-01-05.
Across two trading sessions it preserved session reset, repeat data integrity,
chronological ordering and the frozen WATCH 50 / HOT 10 capacity limits.

Phase 2D replaces range-wide in-memory concatenation with day-partitioned
streaming output. The replay semantics remain unchanged while memory is bounded
to one trading session, which is the scaling prerequisite for longer January
development runs. Request 109 validates the partitioned path on the same frozen
two-session slice before expansion.

Phase 2E/2F now support six-session, day-partitioned market replay with compact
artifacts and causal bar-completion decision timestamps. The first frozen
baseline places 67.36% of +10% crossings in WATCH-or-better before crossing but
only 4.37% in HOT/POSITION, with zero-minute median HOT lead. Historical
one-second aggregates are available under the current data entitlement, while
historical trades/NBBO are not.

The active research task is therefore a preregistered WATCH -> HOT
information-value test: determine whether the just-completed 60-second path at
one-second resolution adds causal near-term urgency information beyond completed
minute bars. This is an attention-layer diagnostic, not an entry-policy or
profitability test.

### Milestone 3 — policy integration

Connect causal attention scoring to SCAN/WATCH, entry/abstention to HOT, and
the recurrent HOLD/EXIT controller to POSITION. Evaluate the full sequence
rather than isolated rows.

### Milestone 4 — execution, capital and safety integration

Add portfolio allocation, empirical quote-aware execution, persistent order
state, degradation monitoring, and external safety controls before any paper
or constrained live deployment.

## Success Criterion

The goal is not to discover a backtest that happened to make money.

Success means building a trader that, using only information genuinely available at each moment, can continuously observe the market, move its attention to developing opportunities, decide when not to trade, enter selectively, adapt its holding duration and exit to evolving conditions, account for realistic execution, and demonstrate repeatable positive economic value on genuinely unseen future periods.

That criterion is the North Star for future MoneyMaker research.
