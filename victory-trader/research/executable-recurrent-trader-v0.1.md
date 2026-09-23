# Executable recurrent trader v0.1 — research design

## Purpose

The attention hierarchy through HOT is now frozen. Request 139 rejected blind
fixed holding, request 140A showed that promotion order matters, and request
140B showed fresh causal learnability of economic opportunity. The next
research object is no longer another scanner score. It is one causal trading
loop over a watched ticker:

    HOT -> ABSTAIN / WAIT / BUY -> POSITION -> HOLD / EXIT

The policy must be allowed to change its mind as new state arrives. A fixed
1/5/15/30-minute holding rule is not an acceptable final policy.

## Evidence carried forward

- Request 138 promoted the market -> active observation -> HOT handoff.
- Request 139 showed first-HOT blind fixed holds are BASE-negative, while a
  minority of episodes contain real post-HOT economic opportunity.
- Request 140A showed first and second HOT promotions are economically stronger
  than third-plus promotions and that HOT score monotonically orders economic
  opportunity.
- Request 140B produced fresh pooled AUC 0.6334 and opportunity-value Spearman
  0.2704; its formal gate failed only because May 15 economic-label coverage
  was 94.59% versus the frozen 95% floor. The result is evidence, not a
  retroactive promotion.

## Architecture to implement

### 1. Episode memory

For every ticker keep causal state across HOT transitions:

- promotion index;
- time since previous HOT promotion;
- time continuously HOT;
- time continuously observed;
- prior HOT-score peak and current drawdown from that peak;
- prior opportunity-score peak and current drawdown;
- whether a position is already open.

This prevents a third or fourth re-promotion from being treated as a brand-new
first opportunity.

### 2. Entry controller

At each score update while not in a position, produce values for:

- ABSTAIN: stop considering the current episode unless rediscovered later;
- WAIT: retain observation and reassess on the next market event;
- BUY: enter at the next causally executable price.

The controller should use predicted economic value rather than an imbalanced
BUY/WAIT/SKIP class label. Historical state-buy-wait-skip work collapsed into
SKIP and is not to be revived.

Training labels may use future outcomes, but input features must be strictly
point-in-time. The target is the incremental BASE value of buying now versus
waiting for the next decision state or abstaining.

### 3. Position controller

After BUY, the action space becomes:

- HOLD;
- EXIT.

At every new decision state compare the estimated value of continuing the
position with the BASE return available from exiting now. Missing state or
invalid execution data must resolve conservatively to an exit-pending state,
not silently extend the position.

Use prior recurrent/optimal-stopping work only as implementation evidence.
Do not reuse its old fitted thresholds or fixed patience rules. Previous work
showed both failure modes: short lookahead can sell too early, while unconstrained
longer-horizon stopping can hold too long.

### 4. Execution and replay

All policy evaluation must use:

- next causally available executable price;
- frozen BASE execution costs as the primary research metric;
- LIGHT and STRESS as sensitivity diagnostics;
- non-overlapping positions per ticker;
- chronological market replay;
- explicit unresolved/missing-fill accounting.

Do not use hindsight oracle exit as realized policy P&L.

## Immediate implementation sequence

1. Build a causal HOT/POSITION state panel that emits one row at every actual
   policy decision event, including promotion memory.
2. Fit development-only action-value models for BUY-vs-WAIT/ABSTAIN and
   HOLD-vs-EXIT.
3. Replay the resulting policy end to end on development/calibration dates.
4. Freeze action rules and economic gates before opening May 21 or later.
5. Run the first untouched executable-policy block only after steps 1-4 pass
   unit/integrity tests.

## Required pre-fresh diagnostics

Before any May 21+ run, verify:

- no future/oracle columns in policy features;
- decision timestamps strictly precede execution timestamps;
- one open position maximum per ticker;
- no overlapping re-entry before exit;
- every realized return is reproducible from raw prices plus frozen execution
  costs;
- promotion memory survives HOT demotion/re-promotion;
- policy behavior is not determined by a fixed holding horizon;
- WAIT actually causes later reassessment rather than becoming implicit BUY;
- missing decision state cannot create optimistic delayed exits.

## Fresh policy gate philosophy

The next fresh gate must evaluate actual executable trades, not oracle
potential. It should require enough trades for interpretation, positive BASE
economic value with uncertainty reported, non-pathological concentration by
day/ticker, bounded severe-loss behavior, and robustness under STRESS costs.

Exact numerical thresholds will be frozen only after the development replay is
implemented and inspected. May 21 and later remain sealed until then.
