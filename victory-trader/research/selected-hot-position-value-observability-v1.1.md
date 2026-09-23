# Selected-HOT position value observability v1.1 — request 144

## Status

Frozen after request 143 diagnosed the request-140B coverage failure and before
inspecting any HOLD/EXIT result from this branch.

Request 142 remains unexecuted because its pre-registration required a formal
request-140B pass. Request 140B did not pass and is not retroactively promoted.

Request 144 is a new, no-new-date diagnostic. It is allowed to reuse the frozen
request-140B selection representation only because every predictive condition
from 140B passed and request 143 established that the sole formal failure came
from sparse post-entry execution references, not predictive ordering.

## Evidence entering this diagnostic

Request 140B on fresh 2026-05-14 through 2026-05-20:

- classifier AUC 0.6334;
- AUC > 0.5 on all five days;
- opportunity-value Spearman 0.2704;
- positive Spearman on all five days;
- frozen selected subset oracle mean +1.255% versus +0.452% overall;
- selected opportunity-positive rate 53.28% versus 42.42% overall;
- formal fail only because 2026-05-15 label coverage was 94.59%.

Request 143 then showed that all 55 missing fresh labels had a valid entry
reference but no later observed minute open inside the 30-minute window. Zero
were missing the entry reference. Fifty-one of 55 had more than 30 minutes
remaining in the session, so this is sparse/illiquid execution behavior rather
than a session-close or timestamp artifact.

## Dates

No new date may be opened.

- fit position paths: 2026-04-30, 05-01, 05-04, 05-05, 05-06;
- calibration position paths: 2026-05-07, 05-08, 05-11, 05-12, 05-13;
- evaluation position paths: 2026-05-14, 05-15, 05-18, 05-19, 05-20.

May 21 and later remain sealed.

## Entry population

The request-140B opportunity classifier, calibration and frozen probability
threshold are recomputed exactly from their original fit/calibration dates.

Report HOLD/EXIT observability for:

1. all first-HOT anchors;
2. the frozen request-140B selected subset.

No opportunity threshold may be changed.

## Explicit anchor path coverage

Request 143 showed that row-level coverage can hide a more important failure:
an entered ticker may produce no usable post-entry decision state.

Therefore report, using the original first-HOT anchor population as the
denominator:

- all evaluation first-HOT anchors;
- anchors that produce at least one causal POSITION decision row;
- selected evaluation anchors;
- selected anchors that produce at least one causal POSITION decision row.

The selected anchor-to-path coverage must be at least 90% for the diagnostic
bridge to pass.

Missing anchors remain execution-risk evidence and are never silently removed
from this coverage calculation.

## Position state

Entry is the next causally executable open after first HOT, under the same
timestamp semantics as requests 139-140.

After entry, emit a state after each completed holding minute, up to the frozen
30-minute research cap. No missing timestamp is compressed.

Causal features remain exactly the request-142 frozen set:

- frozen broad minute/state features;
- frozen trailing one-second features through the decision timestamp;
- current/entry prices and current return;
- running post-entry high/low, peak drawdown and trough recovery;
- minutes held;
- minutes since open and minutes to close.

Future prices and targets are forbidden from model inputs.

## Diagnostic A — one-minute continuation

Target:

    hold_advantage_1m_pct =
        BASE return exiting next exact minute open
        - BASE return exiting now

Model and calibration are unchanged from the request-142 draft:

- HistGradientBoostingClassifier;
- learning_rate 0.05;
- max_iter 180;
- max_leaf_nodes 15;
- min_samples_leaf 75;
- l2_regularization 2.0;
- random_state 20261052;
- chronological Platt calibration on the calibration block.

Semantic HOLD diagnostic: calibrated P(positive one-minute advantage) > 0.5.
No threshold search.

## Diagnostic B — residual remaining option value

Target:

    remaining_option_value =
        best later BASE return through 30m
        - BASE return from exiting now

Remove the fit-only mean remaining value for each integer held minute before
training so the regressor must learn state differences rather than merely
"early minutes have more time left."

Model and calibration are unchanged from request 142:

- HistGradientBoostingRegressor;
- learning_rate 0.05;
- max_iter 180;
- max_leaf_nodes 15;
- min_samples_leaf 75;
- l2_regularization 2.0;
- random_state 20261053;
- fit-only 0.5%/99.5% target winsorization;
- one chronological calibration mean-bias offset.

Semantic patience diagnostic: adjusted predicted excess option value > 0.

## Required diagnostics

For all first-HOT paths and the frozen selected subset report:

- anchor-to-position-path coverage;
- causal decision rows and one-second feature coverage;
- HOLD AUC;
- realized one-minute BASE advantage for semantic HOLD rows;
- day-balanced semantic-HOLD advantage;
- global Spearman for excess remaining option value;
- same-held-minute Spearman median and positive fraction;
- realized excess option value for predicted-positive rows;
- day-balanced realized excess option value;
- all diagnostics by evaluation day.

## Frozen diagnostic bridge

Request 144 passes only if the frozen selected subset satisfies all:

1. selected anchor-to-position-path coverage >= 90%;
2. one-minute HOLD AUC > 0.50;
3. semantic HOLD rows have positive arithmetic incremental BASE mean;
4. semantic HOLD rows have positive day-balanced incremental BASE mean;
5. remaining-value global Spearman > 0;
6. median eligible same-held-minute Spearman > 0;
7. predicted-excess-positive rows have positive day-balanced realized excess;
8. the signs in conditions 2 through 7 are simultaneously positive on at least
   four of five evaluation days with sufficient support.

Passing does not promote an executable trader and does not convert request 140B
into a pass. It only establishes that, once a selected HOT position is open,
the evolving state contains usable information for recurrent HOLD/EXIT.

If it passes, the next development step is to build the full executable
recurrent trader with a separate causal executability/liquidity gate, then
freeze actual BUY/WAIT/ABSTAIN/HOLD/EXIT rules before opening May 21+.

If it fails, do not tune these boundaries on May14-May20.

May 21 and later remain sealed.
