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


### Deferred research branch — anticipatory pre-surge watchlist

Request 134 established that many nominal first +10% crossings are not
recoverable by faster reaction alone: a large fraction of minute-blind names
first become observable only after they have already jumped through the
threshold. The active roadmap therefore does **not** require the current
real-time scanner to predict information that is absent from the live stream.

A separate future research branch is reserved for **anticipatory candidate
discovery before the immediate surge window**. Its purpose is to test whether
longer-horizon, strictly causal signals can place some future runners into a
low-cost WATCH pool before the explosive move begins. Candidate evidence may
include multi-session accumulation/volume structure, liquidity and spread
changes, volatility compression/expansion, repeated abnormal buying pressure,
relative-volume persistence, price structure, corporate/news catalysts, and
other point-in-time information available before the surge.

This branch must be treated as a distinct forecasting problem rather than as a
reason to relabel unobservable jumps as scanner failures. It should complement,
not replace, the broad live scanner:

```text
ANTICIPATORY WATCHLIST  ----\
                            -> DYNAMIC ATTENTION -> HIGH-RES OBSERVATION
LIVE MARKET SCANNER    ----/
```

It is intentionally deferred until the current attention/transport handoff,
HOT observation, ENTRY/ABSTAIN and recurrent position-management path is
validated. When opened, it must use its own chronological training and sealed
forward blocks and must prove incremental economic value rather than merely
higher retrospective runner recall.

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

The targeted one-second reranker passed its frozen information-value gate.
Request 123 then replayed the full learned hierarchy chronologically: it placed
431 of 797 future first +10% runners in HOT strictly before crossing versus zero
for the old percentile runtime. Its formal gate remained false only because the
zero-capture comparator had no defined captured-only lead median. That outcome
is not retroactively promoted.

Request 124 then tested immediate HOT readiness on fresh 2026-02-18 through
2026-02-24 sessions. Learned exact-prior-minute HOT capture was 37.11% versus
zero for the old runtime and passed its 30% floor, while the formal gate failed
because focus admission captured only 38.84% versus the frozen 50% floor. The
top-20 retained 98.81% of focus captures, one-second coverage was 100%, and HOT
occupancy stayed within 10, isolating the bottleneck upstream of those layers.

Request 125 passed every preregistered focus-admission condition on untouched
2026-02-25 through 2026-03-03 sessions. Exact-prior-minute focus capture rose
from 33.63% to 58.63%, improved on all five days, and exceeded the 50% floor.
Market-wide average precision rose from 0.0681 to 0.1469, while the top-20
retained 96.42% of focus captures within its fixed capacity.

Request 126 passed every attention-performance gate on untouched 2026-03-04
through 2026-03-10 sessions. Exact-prior-minute HOT capture rose from 27.18% to
51.16%, improved on all five days, and retained 94.30% of top-20 captures with
100% second-data coverage. However, it required 11,327 one-second ticker-days
and replaced 4.79 of 10 HOT slots per minute, so operational efficiency is not
yet adequate for ENTRY/ABSTAIN integration.

Request 127 preserved exact-prior-minute capture exactly (51.90% for both
persistent and stateless top-20), but failed the operational gate: ticker-day
demand fell only 8.61% and retention reached only 66.04%. The remaining
bottleneck is minute-rank oscillation, not predictive loss.

Request 128 raised mean retention to 76.31% and limited pooled capture loss to
0.92 percentage points, but still failed the operational gate because
ticker-day demand fell only 18.51% rather than 25%. Rank-only persistence has
therefore reached its useful limit.

Request 129 met the resource objective with a 46.35% ticker-day reduction, but
first-come budget allocation collapsed exact-prior capture from 53.56% to
40.17%. The failure is late-session admission starvation, so the first-come
policy is rejected.

Request 130 also rejected the 300-symbol ceiling. Linear release cut demand by
41.74% and raised retention to 80.63%, but lost 8.97 percentage points of
capture and was noninferior on zero of five development days. April remained
sealed.

The v1.4 bridge now passes transport fault tests and accepts score updates at
arbitrary event cadence rather than requiring a one-minute timer. It preserves
selected membership, enforces 20 concurrent subscriptions, and reports churn,
freshness, missing bars and reconnect state.

Request 131 evaluated the complete focus -> event-driven rank-40 shortlist
-> selection-preserving transport -> one-second -> HOT-10 hierarchy on the
first five April sessions. Exact-prior-minute HOT capture improved from 21.58%
to 42.61%; two-minute capture rose from 23.05% to 43.89% and five-minute
capture from 26.63% to 50.14%. Transport preserved model membership exactly,
never exceeded 20 subscriptions, averaged 4.63 additions per decision and
retained 100% one-second coverage. The frozen promotion gate nevertheless
failed because focus-60 prior-minute capture was 49.40% versus the 50% floor
and shortlist capture conditional on focus was 91.82% versus the 95% floor.
HOT conditional retention was 93.93%, so the remaining loss is upstream.

Request 132 decomposed that failure without opening new dates. Only 557 of
1,089 crossings had an eligible exact prior completed-minute hazard row, and
focus-60 captured 538 of those 557 (96.59%). The 49.40% headline focus rate is
therefore near the observability ceiling of the current completed-minute input
on that block rather than evidence that the ranker misses half of observable
opportunities. Stateless top-20 retained 497 / 538 focus captures (92.38%);
rank-40 hysteresis retained 494 / 538 (91.82%), so incumbency cost only three
net captures. The remaining fixed-capacity loss concentrated in broad
opportunity regimes such as 2026-04-08.

Request 133 then tested bounded-turnover challenger admission on untouched
2026-04-09 through 2026-04-15 sessions. It raised shortlist retention from
94.49% to 95.66% of focus captures and was non-lower on all five days, but
failed the operational gate because subscription additions rose from 4.20 to
8.01 per decision. This validates responsive attention while rejecting the
attempt to encode both model preference and transport stability in one set.

Request 134 then measured one-second observability inside the 532
completed-minute blind crossings from the already-opened first April block.
All 532 had second data, but 381 / 531 located second-level crossings (71.75%)
were already beyond +10% on their first active second. Only 150 / 531 (28.25%)
had any active pre-threshold second, only 21 had at least five active seconds
of warning, and only five had at least ten. Event-driven broad scanning remains
architecturally necessary, but most of the old headline recall denominator
contains sparse jump/gap events that provide no causal pre-cross observation
window in the available stream. Scanner evaluation must therefore report both
raw crossing recall and recall conditional on genuine causal observability.

Request 135 tested desired/active separation on untouched 2026-04-16 through
2026-04-22 sessions. The active transport retained 558 of 585 focus captures
(95.04%) versus 556 for rank-40 hysteresis, exceeded the comparator in pooled
capture, and was non-lower on four of five days while respecting 20 active
slots. Its formal gate remained false because the audit averaged 5.004
additions per decision after counting mandatory session initialization against
a rate limit that was preregistered only for post-initial updates. Request 135
was not retroactively promoted.

Request 136 then confirmed the unchanged policy on untouched 2026-04-23 through
2026-04-29 sessions with the operational audit aligned to the already-frozen
initialization exception. Active membership retained 341 of 349 focus captures
(97.71%) versus 337 for rank-40 hysteresis, improved pooled exact-prior capture
from 56.17% to 56.83%, and was non-lower on all five days. Post-initial
subscription additions averaged 4.935 per decision, never exceeded five, and
active occupancy never exceeded 20. The desired/active separation and
rate-limited observation transport were therefore promoted.

Request 137 integrated that transport with the frozen one-second reranker and
HOT-10 on untouched 2026-04-30 through 2026-05-06 sessions. Predictive handoff
quality passed: Focus captured 634/648 causally observable crossings (97.84%),
active observation retained 95.43% of Focus captures, HOT retained 94.88% of
active prior-minute captures, second-data coverage was 100%, and exact-prior
HOT capture improved from 52.07% for the rank-40 comparator to 52.81% with
non-lower performance on all five days. The formal gate nevertheless failed
because a row-reconstructed churn audit reported 5.0005 mean post-initial
additions and a maximum of seven even though the selector hard-caps true
subscription additions at five. The audit was conflating temporary absence of
a scoreable feature row with subscription removal/re-addition. Request 137 is
not retroactively promoted.

Request 138 is the active fresh confirmation on 2026-05-07 through 2026-05-13.
It changes no predictive model, capacity, reranker or HOT rule. Subscription
membership is now recorded directly from the selector, separately from whether
the subscribed ticker has a scoreable feature row at that instant. If this
explicit-state full handoff passes, Milestone 2 attention/observation research
is promoted and the active research frontier moves to HOT -> ENTRY/ABSTAIN.
A conditional first-HOT economic bridge is already preregistered to reuse the
same request-138 dates without opening May 14 or later.

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
