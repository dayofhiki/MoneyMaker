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

Request 138 then passed the explicit-state full handoff on untouched
2026-05-07 through 2026-05-13 sessions. Of 559 causally observable exact-prior
crossings, Focus captured 551 (98.57%). Active observation preserved the
required focus coverage, HOT retained 95.25% of active prior-minute captures,
and active HOT-within-1m was non-lower than the rank-40 integrated comparator
on four of five days. The explicit selector-state audit confirmed a maximum of
20 concurrent subscriptions, mean 4.988 post-initial additions per decision,
maximum five, and zero selection/transport mismatch. The legacy row-reconstructed
audit still falsely reported a maximum of seven additions, confirming the
request-137 diagnosis. Temporary unscoreable subscription rows were 1.72% of
subscription-state rows and did not imply unsubscribe/re-add. Request 138 passed
the frozen promotion gate, so Milestone 2 attention/observation handoff is
promoted.

Request 139 then reused the already-opened request-138 dates to measure
first-HOT economics without opening May 14 or later. Across 2,091 first-HOT
episodes, blind fixed 1/2/5/10/15/30-minute holds were all BASE-negative, with
means between -1.39% and -1.17%, despite mildly positive gross drift. The
non-executable 30-minute best-exit ceiling was +0.339% on average, was
BASE-positive in 40.70% of evaluable episodes, and had a positive mean on every
evaluation day. The median best BASE return remained negative (-0.431%).
Therefore HOT is validated as an attention state, not an entry rule: economic
opportunity is present but sparse enough that most HOT episodes should be
abstained from, and fixed holding time is rejected.

Request 140A then reused the same opened dates and expanded the economic unit
from one first-HOT ticker-day episode to every causal HOT promotion. Across
9,597 promotions, broad promotion entry was weaker than first-HOT entry: the
30-minute oracle BASE mean was -0.104% overall. Promotion order mattered. First
promotions retained +0.339% mean oracle BASE value, second promotions +0.209%,
while third-and-later promotions fell to -0.315%. Re-promotions were separated
by a median seven minutes. The frozen HOT score nevertheless contained real
economic ordering: its Spearman correlation with oracle BASE return was +0.150,
and oracle means rose monotonically from -0.616% in score quintile 1 to +0.685%
in quintile 5. Thus HOT score is informative but is still not an entry rule,
and later ENTRY research must represent promotion order/spacing explicitly.

Request 140B then completed on fresh 2026-05-14, 05-15, 05-18, 05-19
and 05-20 sessions. Causal first-HOT economic opportunity replicated strongly:
pooled classifier AUC was 0.6334 with AUC above 0.5 on all five days, and the
opportunity-value regressor achieved Spearman 0.2704 with positive correlation
on all five days. The calibration-frozen top-quartile score selected 503 of
1,881 labeled episodes; their oracle BASE mean was +1.255% versus +0.452% for
all first-HOT episodes and their cost-positive opportunity rate was 53.28%
versus 42.42%. The formal gate remained false only because 2026-05-15 label
coverage was 94.59%, 0.41 percentage points below the frozen 95% floor. That
gate is not weakened and request 140B is not retroactively promoted.

Request 143 then diagnosed the sole request-140B formal failure without
opening new dates. All 55 missing fresh economic labels had a valid exact
next-minute entry reference but no later observed minute open inside the
30-minute window. Fifty-one of 55 still had more than 30 minutes left in the
regular session, and the missing rows generally had thinner trailing second
activity. The failure is therefore an executability/liquidity problem rather
than an entry-timestamp or session-close artifact. Request 140B remains a
formal fail.

A subsequent implementation audit invalidated request 144 as research
evidence before it produced any artifact. Its runs terminated during the
position calculation, and the audit found two semantic defects in the draft:
multi-day batching could carry prior-minute features across a session boundary,
and the next executable minute open was exposed to the HOLD/EXIT model before
the decision that was supposed to precede that fill. The audit also found that
the historical request-140B top-quartile diagnostic threshold had been computed
only on future-labeled calibration rows, which is unsuitable for an executable
policy. None of these findings retroactively change request-140B's AUC/value
ordering evidence, but they prevent promotion of the old selected subset.

The corrected recurrent path now resets lag state by session, uses only
completed-bar close/path information as POSITION features, keeps next opens
strictly label/execution-only, calibrates the policy selection quantile across
all causal calibration feature rows, and streams one market day at a time.
Second-window search, running extrema, future-label suffix maxima, and selective
ticker-day materialization reduce compute without changing their frozen
mathematical definitions.

Request 145 then completed successfully on the already-opened Apr30-May20
sessions and formally failed its frozen bridge. It reported broad HOLD AUC
0.5541 and remaining-option-value Spearman 0.3026; the policy-selected subset
reported HOLD AUC 0.5343 and remaining-value Spearman 0.2233, but only three of
five sessions had all signs positive and selected anchor-to-path coverage was
89.36% versus the 90% floor.

A second implementation audit subsequently found that request 145 still
conditioned the POSITION decision-row population on a future execution event:
a completed causal state was omitted whenever the following minute lacked an
exit-reference open. Its reported decision-state coverage of 1.0 was therefore
tautological rather than a genuine coverage measurement. The implementation
now keeps every causal completed state and marks the future exit reference and
supervised labels missing when no such fill reference exists. Request 145
remains a formal failure and its HOLD/remaining-value metrics are treated as
conditional diagnostics, not promotable recurrent-policy evidence.

The same audit found a more upstream selection leak in the historical
attention path. `build_market_hazard_rows` required an exact next-minute bar
to construct its supervised target, and historical inference reused that
labelable-only population. Whether a ticker will print in the following minute
is not known at decision time. Although exact-prior runner rows themselves are
labelable, removing future-silent competitors can change cross-sectional
Focus/active/HOT allocation. The code now separates all causal inference rows
from nullable supervised labels. Model fitting may use identifiable labels;
market ranking and attention allocation may not inspect future bar existence.

Request 146 then completed on the already-opened May7-May13 block using
the fully causal inference population. The corrected market population restored
515,850 current rows (22.94% of causal inference rows) that the historical path
had removed based on future next-minute bar existence. Focus quality remained
strong: it captured 547 of 559 causally observable exact-prior crossings
(97.85%). HOT-within-1m remained 49.54%, while HOT-within-2m and 5m rose to
58.58% and 65.36%. The formal gate nevertheless failed because Active retained
only 93.60% of Focus captures versus the frozen 95% floor. Temporary unscoreable
subscription state rose to 6.35% and mean scoreable Active occupancy fell to
18.73 / 20. The remaining upstream bottleneck is therefore not broad discovery
or the one-second HOT reranker; it is allocating scarce high-resolution Active
slots to names that remain causally observable/actionable.

Request 147 tested that bottleneck without increasing Focus, Active, HOT or
transport-churn budgets. A separately fitted causal observability model predicts
whether a current ticker will print an exact next-minute aggregate. Active
priority was frozen as market runner hazard multiplied by predicted next-minute
observability, with new desired Active subscriptions restricted to Focus-60.

The authoritative Focus-first request-147 run 35884196183 completed and failed
the frozen improvement gate. Active retention improved from 93.60% to 94.33%
but remained below the 95% floor. HOT-within-1m / 2m / 5m all improved modestly
to 50.26% / 59.20% / 65.57%, with 100% one-second coverage and all resource
budgets respected. However, aggregate temporary-unscoreable subscription rate
worsened from 6.35% to 6.92% and mean scoreable Active occupancy fell from
18.73 to 18.62. One-minute observability factorization is therefore not
promoted.

This result narrows the next question. Aggregate unscoreable state currently
mixes fundamentally different causes, including a desirable transition into a
runner state and true sparse/no-bar disappearance. Those reasons must be split
before unscoreable occupancy is treated as wasted transport capacity.
In parallel, a direct unconditional next-minute actionable-crossing model can
be compared with the factorized hazard-times-observability score, and a
transport-horizon persistence target can be tested if one-step observability
does not rank sustained scoreability well.

Request 148 remains preregistered but blocked because request 147 did not pass.
It will preserve the request-140B ENTRY feature set and model family if/when a
corrected Active foundation passes. May 21 and later remain sealed.


Request 149 then separated Active admission from stale-incumbent eviction.
Admission kept request-147 actionable priority, while a causal three-minute
transport-survival model ranked stale incumbents for eviction. The historical
mixed unscoreable metric was also decomposed into runner graduation versus
genuine sparse missing-bar waste.

With the frozen five-addition primary budget, the retention policy materially
fixed slot utilization but did not pass the predictive handoff gate. Genuine
sparse-waste fell from 6.38% under the request-147 policy to 1.85%, mean
scoreable Active occupancy rose to 19.60 / 20, and mixed unscoreable state fell
to 1.98%. However Focus -> Active retention was only 93.78%, HOT-within-1m was
49.54%, and the formal gate remained false.

The preregistered non-promotable churn sensitivity isolated the remaining
bottleneck. With max additions 6, Active retention was 94.33% and sparse-waste
1.25%. With 8, retention reached 95.25%, sparse-waste fell to 0.42%, and mean
scoreable occupancy reached 19.91 / 20. Allowing unconstrained convergence up
to 20 additions improved retention only slightly further to 95.43%, showing
strong diminishing returns after roughly eight additions. The old five-name
transport ceiling is therefore a binding research resource constraint rather
than a demonstrated property of the desired final trader.

Request 150 then tested a balanced stale-incumbent value on the same
development block: actionable runner priority multiplied by three-minute
transport survival. It did not solve the five-addition bottleneck. Focus ->
Active retention was 93.60%, genuine sparse waste 2.84%, mean scoreable Active
occupancy 19.39 / 20, and HOT-within-1m / 2m / 5m was
50.57% / 59.20% / 65.98%. The primary gate remained false.

Its frozen sensitivity strengthened the resource-ceiling diagnosis. With the
same balanced retention value and an eight-addition ceiling, Active retention
reached 95.61%, genuine sparse waste fell to 0.45%, and mean scoreable
occupancy reached 19.91 / 20. The unconstrained 20-addition diagnostic retained
95.43%, so the development evidence suggests sharply diminishing benefit above
roughly eight additions. These sensitivity results remain non-promotable.

Request 151 then opened the preregistered fresh block
2026-05-21, 05-22, 05-26, 05-27 and 05-28. Raising the demand-responsive
addition ceiling from five to eight materially repaired transport utilization:
Focus -> Active retention improved from 92.53% to 93.42%, genuine sparse-slot
waste fell from 2.06% to 0.25%, and mean scoreable Active occupancy improved
from 19.55 to 19.95 / 20. The primary nevertheless failed the frozen promotion
gate. It remained below the 95% retention target, pooled HOT-within-1m moved
slightly from 50.99% to 50.81%, and only three of five sessions were non-lower
on HOT-within-1m. The eight-addition ceiling is therefore not promoted from
request 151.

Request 152 reused only those already-opened request-151 sessions and mapped
the frozen Active policy across addition ceilings 5, 8, 10, 12, 16 and 20. It
was diagnostic-only and opened no later date. Retention was 92.53%, 93.42%,
93.12%, 92.97%, 92.83% and 92.83%, respectively. At cap 16 and 20, mean and
minimum scoreable Active occupancy were both 20 / 20 and genuine sparse-slot
waste was 0%, yet Focus -> Active retention remained below 95%. Full minute-level
refresh therefore cannot repair the remaining miss. The preregistered diagnosis
is **selection_or_admission_bottleneck**, not residual transport capacity.
Notably, cap 20 was 0.60 percentage points worse than cap 8, so extra churn is
not merely unnecessary; on this block it displaced useful incumbents.

Request 153 attributed the remaining Active miss and tested a single direct
one-minute crossing model on the same already-opened block. Direct admission
improved cap-20 Focus -> Active retention only from 92.83% to 93.27%
(+0.45pp), below the preregistered 1pp material-gain threshold and below the
95% target. The result was **direct_formulation_insufficient**.

Request 154 decomposed the one-minute admission score family. The Focus-only
direct model materially improved ranking quality (Focus average precision
0.1810 versus 0.1265 for the legacy score) and raised cap-20 retention to
93.57%, but the best cap-20 gain was still only +0.75pp. The preregistered
diagnosis was **feature_information_bottleneck**: changing the one-minute score
formulation alone does not recover enough Active coverage.

Request 155 then changed the target itself from next-minute crossing to
Focus-specific crossing within 1, 2, 3 or 5 wall-clock minutes, with three
minutes preregistered as primary. The primary three-minute score reduced mean
cap-8 post-initial additions from 7.83 to 7.16 per decision while improving
retention from 93.42% to 93.87%. Under cap-20, where the smoother temporal score
can express its ranking without the older transport bottleneck, retention
improved from the legacy 92.83% to 94.17% while mean additions fell from 10.00
to 7.86. Two-minute and three-minute cap-20 retention both reached 94.17%;
five minutes degraded to 93.27%. The primary gate therefore remained false and
the diagnosis was **temporal_target_insufficient**. The experiment nevertheless
confirms that short multi-minute context reduces attention churn and recovers
real coverage; it simply does not yet recover the final ~0.83pp needed to reach
95%.

Request 156 then challenged the fixed Active-20 assumption directly. Using the
same Focus-specific three-minute score, fit-period score quantiles were converted
to frozen absolute thresholds and applied unchanged on the already-opened May
block. The 65th-percentile fit threshold (probability 0.0040281) averaged 19.93
Active names, slightly fewer than Top-20, yet raised Focus -> Active retention
from 94.17% to 95.81%. Active size became genuinely state-dependent: median 18,
minimum 5, p90 31, p95 38 and maximum 60, with more than 20 names on 35.64% of
decisions. Its selected three-minute positive rate was also slightly higher than
Top-20 (4.43% versus 4.23%). The preregistered diagnosis was
**dynamic_threshold_reaches_target_with_lower_burden**. This is strong
development evidence that a rigid 20-name Active budget is itself a bottleneck.

Request 157 then optimized the fixed learned-Focus ceiling while freezing the
request-156 q65 Active threshold. Focus-60 was not near the preregistered
capture frontier: it captured 669/684 supported exact-prior crossings (97.81%),
whereas Top-120 captured 680/684 (99.42%), Top-180 captured 683/684 (99.85%),
and unbounded Focus captured 684/684. The frozen downstream Active policy
captured 641/684 (93.71%) at Focus-60, 644/684 (94.15%) at Focus-120, and
645/684 (94.30%) at Focus-180; unbounded Focus did not improve beyond 645/684.
Under the preregistered rule requiring both Focus capture within 0.25pp of the
best tested frontier and downstream Active capture within 0.50pp of its best
frontier, the smallest qualifying finite cap was **180**, giving diagnosis
**focus_cap_increase_supported**.

This result shows that Focus-60 was a real upstream bottleneck, but also that
downstream Active capture saturates much earlier than raw Focus capture.
Expanding Focus from 60 to 180 recovers 14 additional supported pre-runner
crossings into Focus but only four additional crossings into the frozen Active
policy; mean Active count rises from 19.93 to 22.60 and selected three-minute
positive rate falls from 4.43% to 3.97%. The likely remaining issue is therefore
not simply insufficient Focus width: the request-156 temporal model was trained
only on fit-period Focus-60 and may not rank the newly exposed lower-hazard
population optimally.

The current development candidate is Focus cap 180 plus the frozen request-156
q65 Active threshold, but it is not promoted yet. Before opening later fresh
sessions, decide whether to (a) freeze this finite-cap architecture as-is for
validation or (b) finish the architecture cleanup by testing an absolute/dynamic
Focus gate and/or retraining the temporal Active model on the broader Focus
population. Request 148 remains blocked until the corrected attention handoff
is promoted.

Additional audit fixes now preserve active-subscription age through temporary
unscoreable rows, preserve HOT episode memory through POSITION state, reset
hazard/session history by trading day, use official exchange session bounds
including early closes, retain causal POSITION states when future execution
references are missing, and represent missing oracle outcomes as unknown rather
than negative.

The executable recurrent trader remains the architectural target. The immediate
order of work is now: (1) diagnose request-152 missed Focus crossings and
replace the weak Active admission/selection rule while keeping the validated
transport accounting; (2) freeze that intervention and validate it on later
unopened sessions; (3) only after the corrected attention handoff passes, run
request 148 economic-opportunity revalidation on chronologically later data;
(4) add causal executability/liquidity value to BUY/WAIT/ABSTAIN and rerun
POSITION observability with silent decision states retained; then (5) integrate
the recurrent entry and HOLD/EXIT loop.

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
