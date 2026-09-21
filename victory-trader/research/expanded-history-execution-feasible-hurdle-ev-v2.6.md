# Expanded-history execution-feasible hurdle EV v2.6 — pre-registration

## Status

Pre-registered after v2.5 completed and was retired, before implementation and
before viewing any v2.6 output.

Use only September 2025 through March 2026 development data. April 2026 and
later remain sealed.

## Motivation fixed before results

v2.4 and v2.5 established repeatable positive out-of-month ordering in the
hurdle score, but absolute selected tails were sparse and failed BASE
profitability. v2.5 preserved positive evaluation EV Spearman in all three
months while its affine zero crossing produced 26 / 0 / 8 evaluated primary
trades and BASE means of -0.289% / n/a / -3.089%.

The v2.5 failure rule explicitly forbids more threshold/calibration engineering
on January-March and directs the next intervention to candidate-universe or new
causal-state design.

This branch changes only the candidate universe. It tests whether the existing
v2.4 hurdle decomposition becomes economically useful when it is trained and
executed only at first anchors with a minimum causal execution-feasibility
profile.

## Fixed execution-feasibility gate

The synthetic accounting order unit remains $1,000.

At a ticker-day's first causal state-eligible anchor, require all of:

- dollar_volume_5m >= $100,000;
- transactions_5m >= 50;
- active_minute_fraction_15m >= 0.80.

Rationale fixed before results:

- $100,000 five-minute dollar volume caps the $1,000 order at 1% of observed
  five-minute dollar turnover;
- 50 transactions in five minutes requires at least 10 transactions/minute on
  average;
- 0.80 active-minute fraction requires trading activity in at least 80% of the
  preceding 15-minute window.

These are structural execution-observability floors, not thresholds selected
from realized returns.

The gate is evaluated only at the already-defined first eligible anchor. If that
anchor fails, the ticker-day is skipped. Do not wait for a later state to become
liquid enough.

The same gate is applied before model fitting, chronological calibration, and
evaluation.

## Frozen data and folds

Development months:

- 2025-09
- 2025-10
- 2025-11
- 2025-12
- 2026-01
- 2026-02
- 2026-03

Evaluate only:

- January 2026 using Sep-Dec 2025 strictly-prior history;
- February 2026 using Sep 2025-Jan 2026 strictly-prior history;
- March 2026 using Sep 2025-Feb 2026 strictly-prior history.

Use the existing 80% fit / final 20% chronological calibration split from v2.4.
Persist pre-gate and post-gate anchor counts and prove calibration dates precede
evaluation.

No April 2026+ data, NBBO, new external source, feature subset search, hand
interaction search, cost change, brokerage action, timing rank, HOLD, or SELL
freedom is allowed.

## Frozen model

Inside the gated universe fit the exact v2.4 three-head hurdle model:

1. calibrated P(BASE return > 0);
2. conditional win magnitude;
3. conditional loss magnitude.

Use the exact v2.4 multi-source causal feature frame, model classes,
hyperparameters, winsorization, Platt calibration, magnitude offsets, random
seeds, fixed 15-minute horizon, and existing BASE/light/stress cost scenarios.

Compute:

    hurdle_EV =
        P(win) * E[gain | win, X]
        - (1 - P(win)) * E[loss | loss, X]

No final affine scale calibration is used.

## Executable policies

All policies act only on execution-feasible first anchors and permit at most one
attempt per ticker-day.

### feasible_earliest_15m_cap1

BUY every execution-feasible first anchor.

### feasible_probability_half_15m_cap1

Diagnostic comparator: BUY iff calibrated P(win) >= 0.50.

### feasible_hurdle_ev_15m_cap1 — primary

BUY iff hurdle_EV > 0 percentage points.

The zero threshold is semantic and is not tuned.

## Required diagnostics

For each evaluation month report:

- fit/calibration/evaluation date provenance;
- pre-gate and post-gate anchor counts;
- execution-feasible rate;
- probability AUC/Brier/log loss;
- conditional win/loss magnitude Spearman;
- hurdle-EV Spearman versus realized BASE;
- selected predicted EV versus realized gross/BASE/stress;
- attempts/evaluated/unevaluable and missing reasons;
- BASE mean, day-balanced BASE, p05, severe-loss rate, worst day;
- external-data coverage inside the gated universe;
- pooled trading-day bootstrap;
- frozen position-ledger light/base/stress marked return and accounting
  completeness.

## Development promotion rule

Do not access April 2026 or later unless ALL conditions hold for
feasible_hurdle_ev_15m_cap1:

1. at least 15 evaluated primary trades in January, February and March;
2. arithmetic BASE mean > 0 in every month;
3. day-balanced BASE mean > 0 in every month;
4. day-balanced BASE mean strictly exceeds feasible_earliest_15m_cap1 in every
   month;
5. BASE p05 and STRESS mean are not worse than feasible_earliest_15m_cap1 in any
   month;
6. hurdle-EV Spearman versus realized BASE > 0 in every month;
7. pooled day-cluster bootstrap 95% lower bound > 0;
8. frozen external-data coverage thresholds remain satisfied;
9. BASE position-ledger marked return > 0 and complete_accounting=true in every
   month.

A sparse positive-looking tail, zero-trade month, light-only profit, or missing
outcome cannot satisfy promotion.

## Failure rule

If v2.6 fails, do not tune the three gate thresholds, hurdle zero crossing,
model capacity, costs, horizon, or feature subset on January-March.

Interpret failure as follows:

- if hurdle ordering collapses inside the gated universe, retire this
  execution-feasibility universe and move to genuinely new causal state
  information;
- if ordering remains positive but selected gross alpha is not positive across
  months, candidate/state information remains the bottleneck;
- if selected gross alpha is repeatedly positive but BASE remains negative,
  execution-cost observability/protocol is the bottleneck and the next branch
  should focus on fill/cost identification rather than more signal modeling;
- only after a replicating cost-positive entry process exists may timing,
  HOLD, or SELL freedom be introduced.

No fresh validation month is consumed here.
