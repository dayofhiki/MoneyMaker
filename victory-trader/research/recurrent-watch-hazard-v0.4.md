# Recurrent WATCH transition hazard v0.4 — pre-registration

## Status

Frozen after request 119 failed its preregistered market-relative extension gate
and before any v0.4 evaluation outcomes are inspected.

This experiment changes the **target formulation**. It does not tune the failed
request-119 market-relative feature family. April 2026 and later remain sealed.

## Question

At every causal WATCH decision, can a learned minute-state model rank which
ticker is most likely to experience its **first +10% prior-close crossing at
the very next minute decision** better than the current attention-score
ordering?

This is a recurrent attention-transition problem. The one-minute target is not
a position holding horizon: the model is evaluated again at every completed
minute while the ticker remains in WATCH.

## Frozen chronology

Build causal market-wide sessions from 2026-01-02 through 2026-02-02 inclusive.

- fit: 2026-01-02 through 2026-01-26;
- evaluation: 2026-01-27, 28, 29, 30 and 2026-02-02.

Those five evaluation sessions are untouched by this recurrent hazard model
family at preregistration time.

## Population

Use every row whose frozen attention state is WATCH and whose ticker has not
already crossed the +10% prior-close runner threshold at or before that
decision.

No outcome-dependent sampling is allowed.

For a WATCH decision at time t, the binary target is 1 only when the same
ticker's first +10% crossing occurs at the next regular one-minute decision
timestamp t + 60,000 ms. Otherwise it is 0.

Rows without a next regular-session minute for that ticker are excluded.

## Features

Use only the established causal minute baseline available at the WATCH
decision:

- attention score;
- attention rank;
- return from nominal prior close;
- current completed-minute body return;
- current completed-minute range;
- log current-minute volume;
- log current-minute transaction count;
- one-minute close return;
- one-minute change in prior-close return;
- current / previous-minute volume ratio;
- current / previous-minute transaction ratio.

Add only two state-history variables that are intrinsic to recurrent attention,
not a new market-information family:

- consecutive WATCH/HOT focus age in completed minutes;
- minutes since the most recent transition into WATCH.

No request-119 market-relative extension and no request-115/116 one-second
feature family is included.

## Comparators

Evaluate three score families on exactly the same WATCH rows:

1. current attention score;
2. fixed minute-state HistGradientBoostingClassifier;
3. climatology, the fit-set positive rate, for calibration context only.

Classifier hyperparameters are frozen:

- learning_rate = 0.05
- max_iter = 120
- max_leaf_nodes = 7
- min_samples_leaf = 40
- l2_regularization = 1.0
- random_state = 17

No class weighting, threshold search or probability recalibration is permitted
after evaluation is inspected.

## Finite HOT-budget diagnostic

At each evaluation timestamp, rank the current WATCH rows by score and select
at most the top 10. This emulates the existing HOT compute budget without yet
changing the runtime state machine.

Report for current attention score and learned hazard score:

- next-step crossing capture: fraction of all positive events whose ticker is
  inside that timestamp's top 10;
- top-10 precision: positive rows / selected rows;
- mean positive-event rank.

Also report:

- pooled and per-day average precision (PR-AUC);
- ROC-AUC as a secondary diagnostic;
- Brier score;
- positive event count and WATCH row count;
- fit positive rate;
- top-10 diagnostics pooled and by day.

## Promotion gate

Proceed to a true learned finite-HOT-budget chronological replay only if all are
true:

1. every evaluation session has at least 1,000 eligible WATCH rows and at least
   10 positive next-step crossings;
2. pooled learned PR-AUC exceeds the current attention-score PR-AUC;
3. pooled learned top-10 crossing capture exceeds the current attention-score
   top-10 capture;
4. learned top-10 capture is at least baseline on at least four of five
   evaluation sessions;
5. learned pooled Brier score is lower than climatology Brier;
6. learned mean positive-event rank is lower (better) than baseline.

Passing this gate promotes only the attention-priority model. It does not create
a BUY signal or establish profitability.

If it fails, do not tune the classifier or one-minute transition definition on
these evaluation dates. The next branch must change the state target or causal
information family.
