# Request 190 — fresh 30-minute economic viability revalidation

## Status

Pre-registered after Request 189 failed economically despite a strong fresh
capture-percentile ranking signal, and before implementing or running Request
190.

The current evidence separates two problems:

1. Request 187 proves the Request-178 BUY population contains a robust positive
   30-minute hindsight ceiling.
2. Request 189 proves causal POSITION state can strongly rank relative exit
   quality, but relative quality alone cannot tell whether the episode's best
   exit is economically positive.

Request 190 therefore tests the missing episode-level signal:

**Can the causal original-HOT state identify BUY episodes that will contain a
BASE-positive executable exit somewhere within 30 minutes?**

No new dates are opened and no recurrent policy is changed.

## Data

Training/development history:
- Request-171 BUY-gated fit POSITION episodes;
- Request-171 chronological calibration POSITION episodes.

Fresh evaluation:
- Request-178 BUY-gated Jun 23/24/25/26/29 episodes.

## Causal entry representation

Reuse the Request-184 original-HOT representation:

- causal minute/attention state;
- completed one-second aggregate state;
- deterministic BASE zero-move cost proxy.

No post-entry POSITION state or future execution reference enters model inputs.

## 30-minute labels

For each episode, collect exact executable BASE exit returns from minutes 1
through 30, including the minute-30 cap reference when available.

Define:

    oracle_30m_base_pct = max(executable BASE exits <= 30m)
    cost_coverable = 1[oracle_30m_base_pct > 0]

These are hindsight labels for observability research only.

## Models

Fit on historical fit rows only:

1. one HistGradientBoostingClassifier for cost_coverable;
2. one HistGradientBoostingRegressor for oracle_30m_base_pct.

Frozen settings:
- learning rate 0.05;
- 180 iterations;
- 15 leaves;
- minimum 75 samples per leaf;
- L2 regularization 2.0;
- equal total sample weight per trading day;
- seeds 20261092 / 20261093;
- regressor fit-target winsorization 0.5% / 99.5%.

Chronological calibration:
- classifier threshold = 75th percentile of calibration probability across all
  calibration feature rows;
- regressor may receive only one equal-day-weighted residual mean offset.

No fresh threshold fitting.

## Required diagnostics

Fresh:
- entry-feature and one-second coverage;
- cost-coverable prevalence;
- classifier ROC AUC and average precision;
- regressor Spearman;
- daily AUC/Spearman where defined;
- frozen top-quartile selected rate;
- selected 30m oracle mean/day-balanced mean;
- selected positive-opportunity rate;
- all-BUY 30m oracle mean/day-balanced mean;
- selected-minus-all-BUY uplift;
- trading-day bootstrap of selected oracle mean;
- positive selected-opportunity days.

## Frozen development bridge

Pass only if:

1. entry-state coverage >=90%;
2. one-second coverage >=90%;
3. fresh classifier AUC >=0.60;
4. fresh 30m-value Spearman >=0.15;
5. at least 4/5 days have positive value Spearman where estimable;
6. selected 30m day-balanced oracle mean >0;
7. selected positive-opportunity rate >=60%;
8. selected mean oracle exceeds all-BUY mean by >=+0.50pp;
9. selected bootstrap 95% lower bound >0;
10. selected oracle mean is positive on at least 4/5 days.

A pass does not promote a trading policy. It only licenses composing this
episode-viability head with the already demonstrated Request-189 exit-timing
head in the next development experiment.
