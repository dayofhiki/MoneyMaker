# Expanded-history remaining-option observability v3.9 — pre-registration

## Status

Frozen after recurrent path policy v3.8 request 96 failed its development
bridge, and before implementing or inspecting any v3.9 output.

Use only September 2025 through March 2026 development data. April 2026 and
later remain sealed.

v3.9 is a diagnostic bridge, not a trading policy. It asks two questions that
v3.8 separated cleanly:

1. with the frozen v3.3 opportunity-gated entries, does a cost-positive exit
   opportunity actually exist inside the same 30-minute window often enough
   that better stopping could matter?
2. can the frozen v3.7 causal path-transition state rank the *remaining*
   multi-minute value of waiting, rather than only the next one-minute sign?

No threshold, lag, feature, phase, entry gate, execution cost or holding cap is
optimized in this experiment.

## Motivation fixed before results

v3.8 composed the successful v3.7 one-step HOLD/EXIT classifier into the
trajectory actually reached by a position. The composition failed economically:

- recurrent gross mean was +0.196749%, +0.049116%, and -0.095555% in
  January-March;
- BASE mean was -0.975018%, -1.149194%, and -1.334468%;
- median holding time was one minute in every month;
- the recurrent path beat always-HOLD-to-30m in all three monthly point
  estimates, but the matched improvement was bootstrap-robust only in February.

Thus the one-step signal is useful but myopic. A path can have a weak or
negative next minute while still offering a substantially better exit several
minutes later. Before adding Bellman recursion, v3.9 tests whether that
remaining option value exists and is causally observable.

## Frozen data and state

Reuse exactly:

- v3.2 first-anchor panels from request 84;
- frozen 2025 state panels from request 54 and 2026 state panels from request 8;
- v3.3 opportunity model and strict P(opportunity) > 0.5 gate;
- v3.7 strict past-only fit/calibration/evaluation partitions;
- v3.7 causal feature frame, including the ten current position-path variables;
- exact 1/2/3/5/8/13-minute deltas for all ten path variables;
- no gap compression;
- the existing light/base/stress execution model;
- hard maximum holding time of 30 minutes;
- April 2026+ sealed.

The v3.7 feature set is not changed. Future information is permitted only in
labels and hindsight-oracle diagnostics.

## Frozen entry and feasible exit references

For every frozen opportunity-gated anchor:

- entry reference is the exact anchor+1-minute open;
- if the entry reference is absent or non-positive, record a missing entry;
- candidate exit references are observed positive opens with timestamps
  strictly after entry and no later than entry+30 minutes;
- a missing timestamp is never shifted or compressed into another minute;
- if no candidate exit reference exists, the trade is unresolved for this
  diagnostic.

For each candidate exit, compute full round-trip BASE return using the existing
modeled buy fill at entry and modeled sell fill at that exit.

## Diagnostic A — hindsight best-exit opportunity ceiling

For every valid entry define:

    oracle_best_base_pct =
        max(BASE round-trip return over observed exits through 30m)

and record the holding minute at which that maximum occurs.

This is explicitly non-executable. It is an upper-bound diagnostic only.

Report by month:

- gated attempts, valid entries, oracle-evaluable entries and coverage;
- oracle best BASE arithmetic and day-balanced mean;
- median, p05 and positive-return rate;
- distribution of oracle-best holding minute;
- 10,000-sample trading-day bootstrap of oracle best BASE return.

The oracle diagnostic answers whether the frozen entry population contains
cost-covering movement in principle. It does not authorize a hindsight policy.

## Diagnostic B — remaining option-value target at each decision state

For each v3.7 post-entry decision row with a valid current EXIT reference and at
least one later observed positive open before the 30-minute cap:

1. compute the full-round-trip BASE return if exiting now;
2. compute the best full-round-trip BASE return among later observed opens up to
   the cap;
3. define

    remaining_option_value_pct =
        best_future_base_return_pct - exit_now_base_return_pct

This target measures how much additional BASE return a perfect future stopping
choice could recover relative to exiting now.

Also persist:

- best_future_base_return_pct;
- exact future opens observed;
- exact future slots available from the current minute to the cap;
- future-observed fraction.

Missing future timestamps are not imputed. They reduce observed future
opportunity and are reported as data-quality diagnostics.

## Removing the trivial remaining-horizon effect

Earlier decision minutes mechanically have more future opportunities. To avoid
declaring success merely because the model learns time-since-entry, fit-period
held-minute baselines are frozen separately for each fold.

For every integer completed holding minute 1 through 29:

    baseline_m = mean fit remaining_option_value_pct at minute m

Define the model target:

    excess_remaining_option_value_pct =
        remaining_option_value_pct - baseline_m

The baseline is computed from fit rows only. Calibration and evaluation targets
use the corresponding frozen fit baseline. Evaluation outcomes never define or
alter a baseline.

## Frozen model

Fit one HistGradientBoostingRegressor on the exact v3.7 transition feature frame.

Settings:

- loss = squared_error;
- learning_rate = 0.05;
- max_iter = 180;
- max_leaf_nodes = 15;
- min_samples_leaf = 75;
- l2_regularization = 2.0;
- random_state = 20261045.

Fit targets are winsorized at the 0.5th and 99.5th percentiles calculated on
fit rows only. Calibration and evaluation targets are never clipped.

On the chronological calibration partition, compute one additive mean-bias
offset:

    offset = mean(actual excess target - raw prediction)

Add that fixed offset to evaluation predictions. No slope fit, tail correction,
threshold search, feature selection or model-capacity search is allowed.

## Required observability diagnostics

For every January-March evaluation fold, on the frozen opportunity-gate stratum
report:

- candidate rows and target-evaluable rows;
- target coverage;
- mean future-observed fraction;
- Pearson and Spearman correlation between adjusted prediction and realized
  excess remaining option value;
- per-held-minute Spearman where at least 20 rows are evaluable, plus the median
  of those correlations and fraction positive;
- predicted and realized mean;
- calibration offset and provenance.

For one fixed score-selection diagnostic, select evaluation rows whose adjusted
predicted excess value is > 0. This zero is semantic because the target is
defined as excess over the fit-period same-minute baseline.

Report for that selected set:

- row count and selection rate;
- realized remaining option value;
- realized excess remaining option value;
- day-balanced realized excess value;
- 10,000-sample trading-day cluster bootstrap, deterministic seed 20261045.

This selected set remains a diagnostic and is not executed as a trading policy.

## Frozen bridge decision

v3.9 supports a separately preregistered fitted optimal-stopping experiment only
if both parts below hold.

### A. Opportunity ceiling must exist in every month

For January, February and March separately:

1. oracle-evaluable coverage >= 90% of valid entries;
2. day-balanced oracle-best BASE return > 0;
3. trading-day bootstrap 95% lower bound for oracle-best BASE return > 0.

### B. Remaining option value must be observable in every month

For January, February and March separately:

4. target coverage >= 90% of candidate decision rows with a valid current EXIT;
5. global evaluation Spearman > 0;
6. median eligible per-held-minute Spearman > 0;
7. selected predicted-excess>0 rows have positive day-balanced realized excess
   option value;
8. their trading-day bootstrap 95% lower bound is > 0.

All eight conditions must hold in all three months.

If A fails, the principal bottleneck is the frozen entry population: even a
hindsight-best exit does not reliably overcome BASE friction. The next branch
must redesign causal entry/candidate value rather than further optimize exits.

If A passes but B fails, cost-covering opportunities exist but the current
minute/path-transition information cannot reliably identify when waiting is
valuable. The next branch must add genuinely new causal intraday information
before fitted stopping.

If both A and B pass, do not open April. First preregister a fitted
optimal-stopping/Bellman experiment on the same development months, using the
frozen state and costs and evaluating only reached trajectories.

No v3.9 result may be used to tune the 30-minute cap, fit-baseline definition,
zero diagnostic boundary, lag grid, held-minute subset, phase subset or model
capacity.

## Result — request 97

Authoritative workflow run: `35619699428`. All three strict past-only monthly
jobs and the aggregate job completed successfully. April 2026 and later
remained sealed.

### Hindsight entry opportunity ceiling

| Month | Valid entries | Oracle coverage | Day-balanced best BASE | 95% lower bound | Oracle BASE-positive rate | Mean best hold |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | 616 | 100% | +3.192605% | +2.795954% | 66.88% | 11.40m |
| 2026-02 | 497 | 100% | +2.621688% | +2.280493% | 70.22% | 13.23m |
| 2026-03 | 613 | 100% | +3.114475% | +2.672905% | 64.76% | 13.08m |

The frozen v3.3 gated entry population therefore contains substantial
cost-positive movement inside the 30-minute window in every development month.
v3.8 did not fail because the selected entries had no economically meaningful
exit opportunity at all.

This oracle is non-executable and is used only to establish the ceiling.

### Remaining-option observability

| Month | Target coverage | Global Spearman | Median same-minute Spearman | Positive minute groups | Selected day-balanced excess | Selected bootstrap lower |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | 99.91% | +0.247181 | +0.239748 | 96.55% | +1.260901% | +0.920378% |
| 2026-02 | 99.88% | +0.210845 | +0.205364 | 96.55% | +1.172865% | +0.714959% |
| 2026-03 | 99.84% | +0.232110 | +0.236728 | 100.00% | +1.572089% | +1.153482% |

The result survives the fit-only held-minute baseline correction. Therefore the
signal is not merely the trivial fact that earlier states have more future time
available.

Rows with predicted excess remaining value above the frozen semantic zero
boundary had realized remaining option value around +3.07%, +3.08%, and +3.39%
in January-March. The selected excess over the same-minute fit baseline was
positive with a positive trading-day bootstrap lower bound in every month.

### Decision

All eight preregistered conditions passed in all three months. The aggregate
artifact printed:

`PASS: preregister fitted optimal-stopping experiment`

The evidence now separates the v3.8 failure:

1. the current gated entries do contain enough intraday movement to overcome the
   frozen BASE friction under substantially better exits;
2. the existing causal path-transition state contains replicated information
   about whether better exit opportunities remain;
3. the one-step HOLD classifier was too myopic when recursively composed.

Proceed to a separately preregistered fitted optimal-stopping experiment. Keep
the v3.3 entry gate, v3.7 causal state/lag grid, BASE/LIGHT/STRESS execution
assumptions and 30-minute safety cap frozen. Do not open April 2026+ yet.

