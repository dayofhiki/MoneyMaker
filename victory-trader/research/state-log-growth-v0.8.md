# State Log-Growth Action Value v0.8 — pre-registration

## Status

Pre-registered before implementation and before viewing any v0.8 result.

This experiment uses only the already-seen January-March 2026 development
panels. It must not load or inspect a fresh validation month.

## Motivation

The direct base-net EV policy remains unstable across development months:
January is negative while February and March are positive. State Action Risk
v0.2 showed that a separate severe-loss classifier has no out-of-month ranking
power. State Entry Ranker v0.3 retained relative within-episode timing signal,
but v0.6 and v0.7 showed that hand-written stopping rules and residual fusion do
not convert that signal into a robust absolute entry policy.

The remaining failure is economically asymmetric: large percentage losses hurt
compound capital growth more than equal-size gains repair it. The current direct
EV model is trained on arithmetic percentage return, which does not encode that
asymmetry. This branch changes the learning objective itself rather than adding
another risk threshold or rank threshold.

## Fixed information set

Reuse the State Entry Ranker v0.3 / direct-EV information set unchanged:

- point-in-time state features;
- IWM point-in-time market context already present in the state panels;
- contemporaneous leave-one-out runner breadth;
- existing causal ticker-local lag features;
- current and previous absolute price transforms.

No new data source, news feature, future-dependent feature, risk classifier,
rank threshold, execution assumption, price filter, or fresh validation month is
allowed.

Keep one attempted entry per ticker-day.

## Log-growth target

For each 5-, 10-, and 15-minute action separately, let `r_h` be the realized
base-net percentage return already produced by the fixed execution-cost model.
Define the training target

`g_h = 100 * ln(1 + r_h / 100)`.

This is the percentage-scaled log change in wealth from allocating one unit of
capital to that action. Equal positive and negative arithmetic returns are no
longer treated symmetrically; large losses receive the economically appropriate
compound-growth penalty.

The implementation must fail rather than silently clip or drop any finite
training label with `r_h <= -100%`. No numerical floor may be chosen after
viewing results.

Unlike the arithmetic direct-EV model, do not winsorize `g_h`. The purpose of
this experiment is specifically to retain the downside penalty and compress the
upside naturally through the logarithm.

## Model

For each horizon, use the same causal training rows, chronological 80/20
fit/calibration split, three-minute training subsampling, feature family, and
HistGradientBoostingRegressor capacity as the existing direct-EV model:

- squared-error loss;
- learning rate 0.05;
- 180 boosting iterations;
- max 31 leaf nodes;
- minimum 300 samples per leaf;
- L2 regularization 2.0.

Fit the same bounded affine calibration on the held-out calibration days, but
calibrate predicted log growth to realized log growth rather than arithmetic
return.

No model hyperparameter may be changed using January-March v0.8 outcomes.

## Policies

### earliest_ev_cap1

Unchanged baseline. Enter the first chronological state in a ticker-day whose
highest calibrated direct predicted base-net arithmetic EV across 5/10/15
minutes is at least +0.50%. Use the highest-direct-EV horizon. The first signal
consumes the ticker-day attempt even if unfillable.

### log_growth_cap1

At each state, predict calibrated log growth for 5/10/15 minutes. Enter the first
chronological state in a ticker-day whose highest predicted log-growth value is
at least

`100 * ln(1.005)`

which is exactly the log-growth equivalent of the baseline +0.50% wealth hurdle.
Use the horizon with the highest predicted log growth. The first signal consumes
the ticker-day attempt even if unfillable.

There is no direct-EV confirmation gate, rank gate, severe-loss gate, local-turn
rule, or post-hoc threshold.

## Evaluation

Leave one month out, train on the other two, and repeat for January, February,
and March.

Report for both policies:

- trades, days, and ticker-day episodes;
- gross, base-net, and stress-net arithmetic mean return;
- arithmetic median, p05, severe-loss rate, and worst day;
- day-balanced arithmetic base-net mean;
- realized mean and median log growth;
- day-balanced realized log growth and worst log-growth day;
- action-horizon mix and median entry minute;
- common-episode entry delay and arithmetic/log-growth outcome deltas;
- pooled trading-day-cluster bootstrap intervals for arithmetic return and log
  growth for `log_growth_cap1`.

## Success rule

Do not promote or consume a fresh month unless `log_growth_cap1`:

1. has at least 15 trades in every month;
2. has positive arithmetic base-net mean in every month;
3. has positive day-balanced arithmetic base-net mean in every month;
4. has positive realized mean log growth in every month;
5. has positive day-balanced realized log growth in every month;
6. improves day-balanced realized log growth over `earliest_ev_cap1` in every
   month;
7. does not worsen arithmetic p05 or stress-net mean versus `earliest_ev_cap1`
   in any month;
8. has a pooled trading-day-cluster bootstrap 95% lower bound above zero for
   both arithmetic base-net return and realized log growth.

All eight conditions must pass.

If this exact branch fails, do not tune the log-growth gate, introduce a capital
fraction, clip the log target, or combine log growth with the retired rank/risk
scores using January-March outcomes. Retire the branch or change the information
set / decision formulation.
