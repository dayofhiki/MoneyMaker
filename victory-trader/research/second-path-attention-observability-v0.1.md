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
