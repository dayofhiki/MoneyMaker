# Request 202 — supply barrier-EV executable bridge

## Purpose

Request 201 found a real supply/turnover directional uplift for the runner
barrier (+10% before -3%):

- comparator AUC 0.6159;
- supply AUC 0.6782;
- +0.0622 uplift;
- 6/8 calibration-day AUC wins.

But the top supply-score groups remained barrier-negative. The binary target
was misaligned because it treated stop_first (-3 proxy) and neither (0 proxy)
as the same class.

Request 202 removes that mismatch.

## Data

No new dates.

- train: Request-171 FIT only;
- evaluate action semantics: Request-171 chronological CALIBRATION only;
- Request-178 remains sealed unless this bridge passes.

Reuse the exact Request-201 strictly-prior D-1 supply features and cached
reference data.

## Frozen runner barrier

Only the runner barrier is used:

- stop_first = -3%;
- take_first = +10%;
- neither = 0%;
- ambiguous / entry-unavailable = unscored target;
- 30-minute first-passage window;
- BASE friction, 1-second latency, 5-second entry expiry.

No barrier size is tuned.

## Primary model: direct barrier expected value

Target on filled, non-ambiguous FIT states:

    barrier_value =
        +10  if take_first
         -3  if stop_first
          0  if neither

Train two equal-day HistGradientBoostingRegressor models with the same frozen
capacity as the recent entry models:

1. base comparator: Request-199 no-rich-second feature family;
2. supply primary: comparator + Request-201 eight supply-turnover features.

Train-only 0.5% / 99.5% target winsorization is allowed but is effectively
inactive for the discrete {-3,0,+10} target. No calibration offset or affine
rescaling is allowed.

Semantic action boundary:

    predicted_barrier_EV > 0

No percentile threshold is selected.

## Causal action replay on calibration

Within each ticker/HOT episode:

1. inspect shadow states minute 1 through 10 chronologically;
2. when predicted supply barrier EV first becomes >0, submit the causal entry;
3. if the five-second entry expires with no fill, return to WAIT and allow a
   later positive-EV state to retry;
4. once an entry is admitted, stop searching that episode;
5. if no admitted positive-EV entry occurs, ABSTAIN.

For every admitted state report:
- actual first-passage barrier proxy;
- take / stop / neither / ambiguous status;
- actual full runner risk-policy replay using the same one-second engine.

Comparator:
- earliest executable shadow entry under the same runner risk policy.

## Required diagnostics

Regression:
- base and supply Spearman versus realized barrier proxy;
- per-day Spearman;
- supply-minus-base Spearman.

Supply action:
- signaled / expired / admitted counts;
- selected episode rate;
- first-passage evaluable coverage;
- actual mean and day-balanced barrier proxy;
- take / stop / neither rates;
- positive barrier-proxy days;
- predicted EV mean.

Full runner replay on admitted trades:
- closed coverage;
- mean and day-balanced BASE net return;
- positive rate;
- mean winner / loser / payoff ratio;
- severe-loss rate;
- positive days.

Matched selected-minus-earliest-executable runner replay difference is required.

## Calibration bridge gate

Request 178 may be opened for a frozen Request-202 fresh-development test only
if ALL hold on Request-171 CALIBRATION:

1. at least 30 admitted episodes;
2. selected episode rate between 5% and 50%;
3. first-passage evaluable coverage >= 95%;
4. supply barrier-EV Spearman >= +0.10;
5. selected day-balanced barrier proxy > 0;
6. selected barrier proxy is positive on at least 6/8 days;
7. full runner closed coverage >= 85%;
8. full runner day-balanced BASE net return > 0;
9. full runner mean return is positive on at least 6/8 days;
10. matched day-balanced runner return versus earliest executable entry > 0.

This is a calibration bridge, not final validation.

## Failure rule

If Request 202 fails, do not tune the zero boundary, add percentile gates,
change the +10/-3 barrier, combine the failed premarket branch, or select a
supply feature subset on calibration outcomes.

Interpret failure according to whether supply EV ordering itself transfers:
- if ordering is weak, supply directional information is not sufficient;
- if ordering is useful but semantic-positive selections remain negative, the
  missing edge is richer catalyst/order-flow context rather than thresholding.
