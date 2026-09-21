# Expanded-history one-step continuation observability v3.4 — pre-registration

## Status

Frozen after v3.3 requests 86–87 and before inspecting any v3.4 output. Use
only September 2025 through March 2026 development data. April 2026 and later
remain sealed. This is an observability diagnostic, not a trading policy or a
promotion test.

## Question fixed before results

v3.3 retained an opportunity signal (January–March AUC 0.645, 0.698 and 0.673)
but rejected planned-duration action selection. On gate-positive/raw-Q-positive
states the chosen BASE return was negative in every month while a feasible
hindsight action was positive. The model replaced the old 30-minute bias with
a 1-minute bias and still ordered durations unreliably.

The next structural question is therefore whether genuinely new causal state
observed after entry distinguishes `EXIT now` from `HOLD one more minute`.

## Frozen data and folds

- Reuse the exact v3.2 first-anchor panels from request 84.
- Reuse the frozen 2025 state panels from request 54 and frozen 2026 state
  panels from request 8.
- Reuse v3.3's strict past-only fit/calibration/evaluation day partitions.
- Refit the unchanged v3.3 opportunity head per fold and retain its frozen
  `P(opportunity) > 0.5` evaluation stratum.
- No April 2026+ data, new external source, cost change, post-result feature
  search or threshold search is allowed.

## Timing and label

Conceptually enter at the exact next-minute open after the frozen anchor. A
state bar timestamped `t` is observable only at `t+1 minute`. For follow-up
states after 1–29 completed holding minutes:

- `EXIT now` uses the exact observed open at `t+1 minute`;
- `HOLD` uses the exact observed open at `t+2 minutes`;
- missing exact timestamps are unevaluable and are never replaced by a later
  bar;
- entry cost is sunk; compare BASE modeled sell fills only;
- sell fee cancels because it is identical on both choices;
- target advantage is `(HOLD sell fill / EXIT sell fill - 1) * 100`.

The 1–29 minute window permits a one-minute hold to terminate at the intended
30-minute forced maximum. Halts and other missing observations reduce measured
coverage rather than silently changing the label.

## Causal features and frozen model

At each follow-up time use the then-observable dynamic state, the existing
causal lag sequence (1/2/3/5/8/13 lags), minutes already held and phase
indicators. Merge only anchor fields absent from the follow-up state, preserving
the v3.0 filing-semantic, split, supply and other point-in-time context without
overwriting new dynamic observations.

Fit one HistGradientBoostingClassifier on all fit-partition follow-up examples:
log loss, learning rate 0.05, 180 iterations, 15 leaves, minimum leaf size 75,
L2 2.0 and seed 20261041. Platt-calibrate on the chronological calibration
partition. The frozen diagnostic action is HOLD iff calibrated probability is
strictly above 0.5; otherwise EXIT. No probability threshold will be tuned.

## Required diagnostics

For each January–March evaluation month report both all-anchor and frozen
opportunity-gate strata:

- candidate rows, exact-reference evaluation coverage and missing reasons;
- AUC, Brier, log loss and HOLD base/selection rates;
- always-HOLD incremental mean;
- classifier policy incremental mean, with EXIT assigned zero;
- day-balanced classifier incremental mean;
- the same metrics for held-minute buckets 1–2, 3–5, 6–10, 11–20 and 21–29;
- the same metrics by market phase;
- a 10,000-sample trading-day cluster bootstrap for the opportunity-gate
  overall policy increment;
- strict fold provenance.

Repeated states are diagnostic counterfactual decisions, not an executable
trajectory. Consequently positive results do not establish account-level
profitability.

## Frozen bridge criterion

Proceed to a separately preregistered recurrent one-minute policy only if the
opportunity-gate overall stratum satisfies all five conditions in every one of
January, February and March:

1. AUC above 0.5;
2. classifier incremental mean above zero;
3. day-balanced classifier incremental mean above zero;
4. exact-reference coverage at least 90%;
5. trading-day bootstrap 95% lower bound above zero.

Failure means the observed minute state is still insufficient to time exits;
do not compose or tune a recurrent policy. Passing only permits the next
preregistered experiment, which must add forced 30-minute exit, halt-aware
accounting and sequential dependence. April remains sealed either way.
