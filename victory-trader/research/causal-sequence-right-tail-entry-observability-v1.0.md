# Request 196 — causal sequence right-tail entry observability

## Status

Pre-registered after Request 195 failed its frozen decomposition gate and before
implementing or running Request 196.

Request 195 confirmed that execution drag is highly observable at shadow-entry
states (fresh Spearman 0.822), while remaining gross upside is still weakly
observable (0.095) and reconstructed new-entry surplus ranking remains weak
(0.058). It nevertheless produced a modest economic ranking uplift: high-score
states had higher hindsight delayed-entry opportunity than the full shadow
population.

The next bottleneck is therefore the **future right tail**, not cost estimation.

Request 196 asks whether the current state snapshot is throwing away useful
temporal shape. It explicitly represents the exact causal sequence of the prior
five minute states and tests whether that improves identification of shadow
entries that still have a large cost-adjusted winner available.

No executable entry or exit policy is claimed in this request.

## Data

Historical fit/calibration:
- Request-171 fit and calibration POSITION rows;
- Request-163 historical scan for executable labels and market regime.

Fresh development:
- Request-178 POSITION + scan shards on Jun 23/24/25/26/29.

No new dates are opened.

Use shadow states with minutes_held in [1, 10].

## Labels

Reuse Request-195 delayed-entry economics.

For each shadow state:
- delayed entry = next causally executable open;
- future exits = later exact opens through original HOT + 30 minutes;
- remaining_best_base_pct = hindsight-best BASE return from that delayed entry.

Primary right-tail target:

    strong_winner_3 = 1[remaining_best_base_pct >= +3.0%]

Secondary diagnostic target:

    strong_winner_5 = 1[remaining_best_base_pct >= +5.0%]

The +3% primary threshold is frozen before Request-196 fresh evaluation and is
motivated by Request-194's calibration-selected -3% hard-stop scale. It is not
fit on Request-196 fresh outcomes.

## Sequence representation

Baseline model:
- Request-191 current-state feature family plus Request-195 causal repricing
  descriptors.

Sequence model:
- all baseline features;
- exact 1, 2, 3, 4, and 5 minute causal lags for the following local state
  descriptors when that exact prior minute exists:
  - attention_score, attention_rank;
  - minute_body_return_pct, minute_range_pct;
  - log_minute_volume, log_minute_transactions;
  - entry_to_current_close_pct;
  - running_max_return_pct, running_min_return_pct;
  - drawdown_from_peak_pct, recovery_from_trough_pct;
  - sec_last5_return_pct, sec_last10_return_pct, sec_last15_return_pct;
  - sec_accel_5_pct, sec_accel_10_pct, sec_accel_15_pct;
  - sec_realized_vol_pct;
  - sec_return_efficiency_60;
  - sec_sign_flip_rate_60;
  - sec_close_location_60;
  - sec_close_vs_vwap_pct;
  - sec_volume_burst_10;
  - sec_transactions_burst_10.

A lag is missing rather than backfilled if the exact prior-minute state is not
observed. Future rows are never used.

## Models

Train two equal-day-weighted HGB classifiers for strong_winner_3:
1. baseline current-state classifier;
2. sequence classifier.

Settings:
- learning rate 0.05;
- 180 iterations;
- max leaves 15;
- min leaf 75;
- L2 2.0;
- seeds 20261100 baseline and 20261101 sequence.

Train a secondary sequence classifier for strong_winner_5 with seed 20261102.

No probability calibration or action threshold is used for the primary gate.
The experiment evaluates ranking observability.

## Fresh diagnostics

For baseline and sequence strong_winner_3 models:
- ROC AUC;
- average precision;
- fresh prevalence.

For the sequence score additionally report:
- per-day AUC;
- AUC improvement over baseline;
- top-quartile-by-score diagnostic on fresh states:
  - strong-winner prevalence;
  - actual remaining_best_base_pct mean;
  - uplift versus all shadow states;
  - per-day actual mean and positive day count.

The fresh top quartile is a **ranking diagnostic only**, not an executable
threshold.

Secondary strong_winner_5:
- AUC and average precision only.

## Frozen development gate

Pass only if all hold:

1. sequence strong_winner_3 AUC >=0.60;
2. sequence AUC exceeds baseline AUC by >=+0.03;
3. sequence AUC is >0.50 on at least 4/5 fresh days;
4. top-quartile strong_winner_3 prevalence exceeds all-state prevalence by
   >=+0.10 absolute;
5. top-quartile actual remaining_best_base_pct exceeds all-state mean by
   >=+0.50pp;
6. top-quartile actual remaining_best_base_pct mean is positive on at least
   4/5 fresh days.

A pass licenses Request 197: a calibration-frozen WAIT / ENTER / ABSTAIN entry
gate using the sequence winner score, combined with the Request-194 hard-stop
risk layer and recurrent post-entry information.

A failure means the last-five-minute aggregate state sequence still does not
contain enough information to identify the profitable right tail. The next
change must then target a materially richer event/second-level sequence model or
a different entry population, not threshold tuning.
