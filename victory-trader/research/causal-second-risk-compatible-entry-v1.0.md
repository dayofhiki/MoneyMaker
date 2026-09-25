# Request 197 — causal one-second risk-compatible entry bridge

## Purpose

Request 196/196A asked whether a future right-tail opportunity exists. That is
not the same as a trade the frozen risk engine can actually capture.

Request 197 labels each causal 1-10 minute shadow state by replaying the new
one-second execution contract. The model therefore learns the economics of the
same stop / partial-take / trailing policy that is later evaluated.

This is still development research. It opens no new dates and cannot promote
MoneyMaker.

## Data split

- train: Request-171 FIT only;
- threshold/rule selection: Request-171 chronological CALIBRATION only;
- development evaluation: already-open Request-178 Jun 23/24/25/26/29 only.

Request-178 may diagnose implementation and economics, never promotion.

## Causal state and execution

Candidate states are completed POSITION shadow states at minutes 1 through 10,
with at least 31 regular-session minutes remaining.

At state time t:
- features may use only information completed before t;
- entry decision is known at t;
- base execution latency is 1 second;
- entry expires after 5 seconds if no active second open exists;
- stop/take/trailing triggers use completed one-second OHLC;
- fills use the first later eligible one-second open after the trigger and
  latency;
- missing seconds never create actions;
- halt intervals block fills and a resume gap is realized at the resume open;
- ambiguous within-second barrier ordering is unscored, not guessed.

Base synthetic friction is used for the model target. This is not a claim about
real broker fills.

## Frozen risk-rule family

Only two policies are allowed:

1. tight: stop 3%, take 5% on 50%, proportional trailing retrace 5%;
2. runner: stop 3%, take 10% on 50%, proportional trailing retrace 7%.

Both retain the 30-minute research safety cap.

No other stop/take/trail rule may be added after calibration results are read.

## Target and models

For each risk rule:
- replay one share from the candidate state;
- target = final base-friction net return if the trajectory closes causally;
- ambiguous, entry-unavailable and unresolved paths remain explicit missing
  outcomes and count against coverage.

Train one equal-day HistGradientBoostingRegressor per risk rule on FIT closed
outcomes only. Winsorization is learned on FIT only.

No calibration rows enter model fitting.

## Calibration-only action selection

For each rule, score all calibration states. Candidate ENTER thresholds are
frozen score quantiles P80 / P90 / P95 measured on calibration only.

For each (rule, threshold), replay the executable decision policy:
- observe states chronologically within an episode;
- submit ENTER at the first score meeting the threshold;
- if that entry order expires without an eligible open, no position exists:
  return to WAIT and permit a later qualifying state to submit a new order;
- once an entry is admitted, never submit another entry for that episode;
- otherwise WAIT through minute 10 and then ABSTAIN.

A candidate must have:
- selected episode rate 5% to 50%;
- at least 30 entered calibration episodes;
- closed-outcome coverage >= 90%.

Choose the eligible pair with highest calibration day-balanced realized net
return, tie-breaking by lower severe-loss rate, then lower entry rate.

The exact numerical threshold is then frozen.

## Fresh development report

On Request-178 report:
- entered episodes and selection rate;
- closed / unresolved / ambiguous / unavailable coverage;
- mean and day-balanced net return;
- positive-trade rate;
- average winner / average loser / payoff ratio;
- severe loss <= -5%;
- positive-return days;
- daily-bootstrap interval;
- entry shadow-minute distribution.

Comparator: first available minute-1 entry under the same selected risk rule.

Matched selected-minus-baseline differences are required where both outcomes
close.

## Development gate

Pass only if:
1. selected rate is 5% to 50%;
2. selected closed coverage >= 90%;
3. selected day-balanced net return > 0;
4. daily bootstrap lower bound > 0;
5. at least 4/5 fresh days have positive mean return;
6. matched day-balanced uplift over minute-1 baseline > 0;
7. selected severe-loss rate is below the baseline.

A pass licenses portfolio integration and latency/friction sensitivity on the
same development block. It does not license untouched-date validation yet.


## Pre-result integrity amendment

Before Request 197 produced any outcome result, the already-completed Work
portfolio probe showed a high rate of entry-order expiry on illiquid HOT
episodes. The initial Request-197 implementation incorrectly ended an episode
after a qualifying ENTER signal even when no entry fill occurred.

That is inconsistent with the continuous trader objective. The implementation
was amended before reading Request-197 outcomes: an expired entry returns to
WAIT and may retry at a later qualifying shadow state. Closed, unresolved, or
ambiguous post-entry paths still terminate the entry search because capital was
actually admitted. The baseline is likewise the earliest executable admitted
shadow entry, not an unavailable minute-1 order.
