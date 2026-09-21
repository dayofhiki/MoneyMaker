# Second-path attention observability v0.1 — pre-registration

## Status

Frozen before second-path outcomes are inspected. Development-only.

This experiment improves the **WATCH -> HOT attention component** of the final
trader. It does not fit an entry policy, alter capital allocation, open April
2026 or later, tune the existing attention thresholds, or claim profitability.

It may execute only after Phase 2F confirms that minute decisions are timestamped
at bar completion rather than bar start.

## Question

When a ticker is already in WATCH, does the causal shape of the just-completed
60 seconds contain incremental information about near-term upside urgency beyond
the information present in completed minute bars?

The current baseline score is intentionally crude: cross-sectional percentile
of return from prior close. Six-session replay showed useful WATCH coverage but
HOT promotion was usually too late. Historical one-second aggregates are
available under the current Massive entitlement, while historical trades and
NBBO are not.

## Frozen development window

Use only the already-open January development sessions:

- 2026-01-02 through 2026-01-09 inclusive;
- first four trading sessions are fit;
- final two trading sessions are evaluation;
- no later date is touched.

The chronological split is fixed before feature outcomes are inspected.

## Sampling

For each session:

1. build the exact causal market-wide scan and attention replay using the
   Phase-2F bar-completion timestamp convention;
2. consider ticker-days with at least one pre-crossing WATCH row;
3. choose 12 ticker-days using ascending SHA256 of
   `trading_day|ticker`; this deterministic selection does not inspect any
   future outcome;
4. retain at most the first 8 eligible WATCH decision rows for each selected
   ticker-day;
5. a WATCH row is eligible only before that ticker has already reached the
   existing +10% prior-close runner threshold.

This bounds historical one-second API demand while preserving a
future-outcome-independent sample.

## Causal second-bar window

For a WATCH decision at `decision_t`, use only one-second aggregate bars whose
one-second interval has fully completed by that decision:

`decision_t - 60s <= second_bar_start`

and

`second_bar_start + 1s <= decision_t`.

No bar beginning at or after the decision can become a feature.

The historical second-bar endpoint is queried once per sampled ticker-day and
then sliced locally for all eligible decisions.

## Primary target

For each WATCH decision, calculate:

`future_max_3m_return_pct`

as the maximum completed-minute close return relative to the current completed
minute close over timestamps strictly after the decision and no later than
three minutes after it.

The three-minute window is an attention-urgency diagnostic, not a final
position holding horizon. It is frozen because the current causal attention
baseline has a three-minute pooled median WATCH lead and the HOT layer must
decide whether additional compute should be allocated before an imminent move.

## Minute-only baseline features

Use only information known at the decision:

- current attention score;
- current attention rank;
- return from nominal prior close;
- current minute body return;
- current minute high-low range;
- log current minute volume;
- log current minute transaction count;
- one-minute close return;
- one-minute change in prior-close return;
- current/previous minute volume ratio;
- current/previous minute transaction ratio.

## Incremental one-second features

Add only features describing intra-minute shape or activity timing:

- active-second count over 60s and 10s;
- age of the most recent completed active second;
- last-10s and previous-10s returns;
- 10s return acceleration;
- first-30s and last-30s returns;
- 30s return acceleration;
- realized second-close volatility;
- positive second-return fraction;
- maximum intra-minute drawdown;
- maximum intra-minute run-up;
- last-10s share of minute volume;
- last-10s share of minute transactions;
- last-10s versus previous-10s volume burst ratio;
- last-10s versus previous-10s transaction burst ratio.

Missing sparse-second measurements remain missing. Do not synthesize trades or
forward-fill nonexistent one-second bars.

## Frozen model

Fit the same fixed
`HistGradientBoostingRegressor` twice:

- baseline: minute-only features;
- extended: minute-only plus one-second features.

Hyperparameters:

- learning_rate = 0.05
- max_iter = 120
- max_leaf_nodes = 7
- min_samples_leaf = 20
- l2_regularization = 1.0
- random_state = 17

No hyperparameter, feature, sample, horizon or threshold search is permitted
after evaluation is seen.

## Evaluation

Report on the final two sessions, pooled and separately by day:

- row count and ticker count;
- target mean/median/p75/p90;
- baseline and extended Spearman rank correlation;
- baseline and extended MAE;
- realized target mean in each model's top prediction quartile;
- incremental extended-minus-baseline Spearman.

Also report second-data coverage and API request counts.

## Decision rule

Treat one-second path information as worth carrying into a richer attention
model only if:

1. both evaluation sessions have at least 30 eligible rows;
2. pooled extended Spearman exceeds pooled baseline Spearman;
3. extended Spearman is not lower than baseline on either evaluation session;
4. pooled extended top-quartile realized target mean is at least as high as the
   baseline top-quartile mean.

This is only an information-value gate. Passing it does not promote a trading
policy or justify opening sealed validation periods.

If the gate fails, do not tune the three-minute horizon, attention thresholds,
sample selection or model hyperparameters against these same evaluation days.
The next branch must use a genuinely different causal information family or
target formulation.


## Result — request 115

Request 115 completed successfully and preserved the frozen causal protocol.

Data audit:

- 72 historical one-second REST requests, zero retries;
- 72/72 sampled ticker-days returned non-empty one-second aggregates;
- 234 eligible WATCH decision rows across 58 tickers;
- one-second feature coverage: 100%;
- median active seconds in the causal 60-second window: 8;
- fit rows: 148 on 2026-01-02 through 2026-01-07 trading sessions;
- evaluation rows: 86 on 2026-01-08 and 2026-01-09.

Pooled evaluation:

| Metric | Minute baseline | + one-second path |
| --- | ---: | ---: |
| Spearman | 0.032468 | 0.051782 |
| MAE | 1.664410 | 1.627909 |
| top prediction quartile realized 3m max return | +0.483345% | +1.038843% |

Incremental pooled Spearman was +0.019314.

By evaluation day:

- 2026-01-08: Spearman 0.079989 -> 0.070897, incremental -0.009091;
  MAE 1.461207 -> 1.411541; top-quartile realized target
  +1.132487% -> +1.348525%.
- 2026-01-09: Spearman -0.082326 -> 0.003163, incremental +0.085489;
  MAE 1.877290 -> 1.854580; top-quartile realized target
  -0.242064% -> +0.078662%.

The preregistered information-value gate **fails** because extended Spearman was
lower than baseline on 2026-01-08, even though pooled ranking, pooled MAE,
pooled top-quartile outcome and all top-quartile day comparisons improved.

## Interpretation

The experiment does not justify promoting the frozen 60-second feature family
into the attention policy. It also does not support the stronger conclusion
that one-second paths are useless: the point estimates are directionally
positive on several diagnostics but not stable across both evaluation days.

Do not tune the three-minute horizon, feature list, sample count, model
hyperparameters or attention thresholds against these evaluation results.

Per the preregistered failure branch, the next experiment must change either
the causal information family or the target formulation. The next target
should better match the final stateful trader than a fixed three-minute
forecast and should use later, untouched January development sessions for
evaluation.


## Chosen next branch — discrete attention hazard

Request 115 does not justify abandoning one-second information. Its pooled
point estimates improved across rank correlation, MAE and top-quartile outcome,
but the improvement was not stable by day and the absolute rank correlations
remained weak.

The next experiment will therefore change the target formulation rather than
tune request-115 thresholds, horizon, feature list or hyperparameters.

The planned target is a repeated discrete-time hazard for the existing runner
event: while a ticker is in WATCH and has not yet crossed +10% from nominal
prior close, estimate whether its **first +10% crossing occurs before the next
minute decision**. This is an attention-state transition target, not a fixed
holding-horizon return target.

The experiment should compare:

1. the current return-rank HOT ordering;
2. a minute-only hazard model;
3. the same hazard model plus the frozen request-115 one-second feature family.

Use the already-consumed 2026-01-02 through 2026-01-09 sessions for fitting and
later untouched January development sessions for evaluation. Expand the
outcome-independent sample size in the new preregistration so the evaluation is
not dominated by tens of rows. Operational evaluation should emphasize
next-step runner-event capture under the finite HOT budget, PR-AUC and
probability calibration, with per-day reporting.

If the one-second hazard model is consistently better, it becomes a candidate
HOT-priority signal and a later calibration phase can allow HOT capacity to
remain partially empty when absolute urgency is low. If it is not better, the
next branch should change the causal information family rather than retune the
same second-path features.
