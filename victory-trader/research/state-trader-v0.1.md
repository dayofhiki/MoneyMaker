# State Trader Research v0.1

## Goal

Replace one-shot threshold-chasing with a continuously observed intraday trader.

The research unit is a **runner state at a specific minute**, not a single threshold event.
A production policy may repeatedly enter, hold, reduce, exit, and re-enter while the
same ticker evolves through different states.

The objective is not raw backtest return alone. Promotion requires positive
after-cost expectancy with controlled drawdown/tail risk, realistic execution,
and explicit edge-decay monitoring.

## Development / validation discipline

- Development months currently seen: 2026-01, 2026-02, 2026-03.
- Do not use 2026-05 through 2026-06 validation or 2026-07 through 2026-08 final holdout.
- Do not consume 2025-12 as an external test until a concrete state policy is frozen.
- State discovery may use January-March because they are already development data.
- Any rule designed after seeing January-March must be tested on a new untouched
  development month before validation.

## Episode definition

A runner episode begins at the first regular-session close crossing +10% versus
the prior close. From that minute until the regular-session close, every observed
minute bar becomes a candidate state.

A hypothetical new long action at state minute t enters only at the exact
next-minute open. Missing exact entry or exact target bars remain missing.

## Point-in-time state representation v0.1

Price/path:
- return from prior close
- minutes since first +10% cross
- HOD distance / running HOD return
- running low rebound since +10%
- minutes since HOD
- new-HOD flag
- exact trailing returns: 1/3/5/10/15m
- 5m and 15m realized volatility
- current bar range/body/close location
- 3m vs 15m range contraction

Structure:
- regular-session VWAP distance
- above-VWAP flag
- prior-5m-high reclaim after a >=1.5% pullback
- pullback depth from HOD
- recent deepest pullback
- coarse descriptive phase tag: impulse, reclaim, pullback-above-VWAP,
  pullback-below-VWAP, consolidation, failure, other

Flow/liquidity:
- 1m/3m/5m volume
- 1m and 5m volume acceleration vs prior 20m
- 3m volume relative to its post-cross peak
- transactions 1m/5m
- 5m dollar volume
- 15m active-minute fraction

Context:
- runners +10% so far today
- runners +10% in last 30m
- IWM return since open / exact 15m return / 15m volatility
- selected safe features already known at the +10% crossing, such as historical
  RVOL and premarket context when available

## Outcomes

For every state, measure hypothetical exact next-minute-open entry and:
- gross return at 1/2/5/10/15/30m
- light/base/stress friction-adjusted return
- 15m MFE and MAE

These are **action-value research labels**, not a deployable strategy.

## Pre-specified state hypotheses for the first screen

The first screen is deliberately small and interpretable:

1. Reclaim after pullback:
   prior 15m reached at least 1.5% below HOD, then close reclaims the prior 5m high.
2. Constructive pullback:
   HOD drawdown between 1% and 4%, still above VWAP, with 3m volume <= 60% of its
   post-cross peak.
3. Fresh HOD impulse:
   new HOD with positive 3m return and above VWAP.
4. Tight consolidation:
   within 3% of HOD, above VWAP, 3m/15m range ratio <= 0.65.
5. Failure state:
   at least 5% below HOD and below VWAP.

The first pass compares these states across January, February, and March and does
not optimize their thresholds after looking at the results.

## Next stage after state discovery

If one or more states show stable after-cost improvement, freeze a simple
position-aware policy:
WAIT / BUY / HOLD / EXIT / RE-ENTER.

Only then use a fresh development month. More flexible statistical/ML policies
come after the state panel and simple policy establish that the signal survives
realistic costs and repeated-trade accounting.
