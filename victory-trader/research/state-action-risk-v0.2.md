# State Action Risk v0.2

## Status

**Retire the severe-loss classifier and the 20% risk gate.**

Keep the causal sequence feature family for further research. Keep one entry per
ticker-day only as a risk-control hypothesis, not as a validated strategy. Do not
promote any policy in this experiment and do not consume a fresh month.

## Purpose

This experiment followed State Action Value Sequence v0.1. It tested whether the
sequence model's after-cost selection could be made robust by:

- fixing the entry EV gate at 0.50%;
- limiting exposure to one entry per ticker-day;
- fitting a train-only classifier for a realized base-net loss of at least 5%;
- rejecting entries with predicted severe-loss probability above 20%.

Each month was held out in turn and models were trained only on the other two
months. The experiment reused January-March 2026 development data.

## Policies

- `repeat_no_risk`: every state with predicted best base EV >= 0.50%;
- `cap1_no_risk`: earliest qualifying state per ticker-day;
- `cap1_risk20`: the same cap, plus predicted severe-loss probability <= 20%.

All policies use the 36 causal ticker-local lag features from Sequence v0.1 and
choose among 5-, 10-, and 15-minute actions.

## Results

| Month | Policy | Trades | Base mean | Day-balanced | Severe-loss rate | Stress mean |
|---|---|---:|---:|---:|---:|---:|
| 2026-01 | repeat_no_risk | 25 | -3.040% | -2.848% | 40.0% | -5.153% |
| 2026-01 | cap1_no_risk | 18 | -2.737% | -2.521% | 38.9% | -4.911% |
| 2026-01 | cap1_risk20 | 11 | -3.159% | -1.651% | 36.4% | -5.244% |
| 2026-02 | repeat_no_risk | 87 | +0.485% | +1.211% | 23.0% | -1.542% |
| 2026-02 | cap1_no_risk | 35 | +2.833% | +4.673% | 20.0% | +0.727% |
| 2026-02 | cap1_risk20 | 20 | -2.018% | -2.358% | 30.0% | -4.005% |
| 2026-03 | repeat_no_risk | 32 | +0.871% | -0.547% | 12.5% | -1.219% |
| 2026-03 | cap1_no_risk | 23 | +0.535% | -0.984% | 13.0% | -1.477% |
| 2026-03 | cap1_risk20 | 13 | +2.428% | +1.616% | 15.4% | +0.277% |

Cross-month, `cap1_no_risk` had two of three base-positive months but only one
day-balanced-positive month. Its median monthly base mean was +0.535%, worst month
was -2.737%, and median stress mean was -1.477%.

The 20% risk gate reduced the sample from 76 to 44 trades. Only February retained
the preregistered minimum of 15 trades in the cross-month summary, and that month's
base mean deteriorated from +2.833% to -2.018%.

## Risk-score diagnostic

The diagnostic uses all 144 `repeat_no_risk` trades and defines a severe loss as
realized base-net return <= -5%.

- overall ROC AUC: **0.473**;
- ticker-day cluster-bootstrap 95% interval: **[0.372, 0.558]**;
- monthly AUCs: January 0.573, February 0.494, March 0.357;
- below the 20% predicted-risk gate: 79 trades, 26.6% actual severe losses;
- at or above the gate: 65 trades, 20.0% actual severe losses;
- highest predicted-risk quintile: 13.8% actual severe losses and +3.678% base mean;
- one-entry-per-ticker-day pool AUC: **0.405**.

The risk score has no out-of-month ranking power. Its highest predicted-risk
quintile was actually the best realized-return group, so threshold retuning on
these same months would be post-hoc inversion rather than validation.

## Uncertainty

Day-cluster bootstrap intervals remain wide and cross zero:

| Policy | Trades | Days | Day-balanced mean | 95% interval |
|---|---:|---:|---:|---:|
| repeat_no_risk | 144 | 44 | -0.351% | [-3.130%, +2.001%] |
| cap1_no_risk | 76 | 44 | +0.981% | [-2.490%, +4.577%] |
| cap1_risk20 | 44 | 31 | -0.834% | [-3.028%, +1.301%] |

The cap improves concentration and the pooled point estimate, but evidence is not
strong enough for promotion.

## Decision

1. Retire `cap1_risk20` and do not tune its threshold on January-March.
2. Retain the 36 causal lags because they materially improved the direct
   action-value baseline.
3. Retain `cap1_no_risk` only as a preregistered exposure constraint in any next
   sequence experiment.
4. Do not revisit a dynamic stop/model-exit branch without new information. Dynamic
   State Policy v0.3.1 already tested a 5% hard stop, structural/model exits, and
   minimum-hold hysteresis; at the 0.50% entry gate it produced only one of three
   base-positive months and no account-positive months.
5. Do not consume a fresh validation month.

## Next research direction

The oracle shows a genuine after-cost timing ceiling (15-minute/180-minute oracle
base mean at least +2.698% in every development month), while fixed state
regression, dynamic exits, and the severe-loss classifier cannot rank that
opportunity reliably.

The next branch should therefore target **relative entry-time ranking within each
ticker-day**, not another absolute return/risk threshold. Pre-register a pairwise
or episode-normalized ranking target, keep causal sequence inputs and the one-entry
cap, compare it directly with `cap1_no_risk`, and require improvement in
day-balanced return and tail metrics across all three development months before
using any fresh data.

## Reproducibility

- request: 24, operation `state_action_risk`;
- workflow run: https://github.com/dayofhiki/MoneyMaker/actions/runs/35345823035
- artifact: `moneymaker-state-action-risk-v02-24`;
- artifact digest:
  `sha256:ca57e00f3bfaca1301690ab90d837b4decf86155060acd2687db350dc347f3af`.
