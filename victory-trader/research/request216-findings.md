# Request 216 findings — event-time recurrent HOLD/EXIT

## Verdict

Request216 **fails the economic and HOLD-value gates**. Event-time POSITION replay improves accounting and slightly reduces the Request212 error, but it does not recover a usable HOLD/EXIT signal.

This is important negative evidence: exact-minute POSITION semantics were a secondary distortion, not the main reason the recurrent position policy failed.

## Canonical run

- Workflow run: `36168163798`
- Artifact: `10879685576`
- Development dates: 2026-06-23, 24, 25, 26, 29
- New dates opened: **no**
- Support entry policy: Request208, used only to create nonempty structural HOLD/EXIT trajectories
- Promotion eligible: **no**

## Event-time coverage

- support entries: **169**
- event-state rows: **4,175**
- evaluable next-event HOLD targets: **3,658 / 4,175 = 87.62%**
- position start-state coverage: **100%**
- completed dynamic trajectories: **94.67%**
- rows arriving after a silent interval: **185**
- previous observed-event gap mean: **1.081 min**
- previous observed-event gap p90: **1.0 min**

The replay is sufficiently complete to evaluate the hypothesis.

## HOLD-value result

Request212 exact-minute reference:

- HOLD advantage Spearman: **-0.01227**
- selected realized HOLD advantage mean: **-0.05521%**

Request216 event-time result:

- HOLD advantage Spearman: **-0.00424**
- selected realized HOLD advantage mean: **-0.04192%**
- selected day-balanced HOLD advantage mean: **-0.03298%**
- predicted HOLD rate: **32.31%**
- good days: **2 / 5**

The sign remains wrong. The event-time formulation reduces the error only slightly.

By day:

- 2026-06-23: Spearman -0.0294, selected advantage -0.0278%
- 2026-06-24: Spearman -0.0463, selected advantage -0.0781%
- 2026-06-25: Spearman -0.0435, selected advantage -0.0982%
- 2026-06-26: Spearman +0.0622, selected advantage +0.0131%
- 2026-06-29: Spearman +0.0474, selected advantage +0.0261%

Only two of five days have the correct local sign.

## Trading result

Event-time dynamic HOLD/EXIT:

- resolved trades: **160**
- mean BASE: **-1.24520%**
- day-balanced BASE: **-1.19788%**
- positive rate: **20.63%**
- p05: **-4.2572%**

Frozen Request208 fixed-3m comparator:

- trades: **169**
- mean BASE: **-1.18859%**
- day-balanced BASE: **-1.11426%**
- positive rate: **18.34%**

Matched event-time dynamic minus fixed-3m:

- mean difference: **-0.05918pp**
- day-balanced difference: **-0.07114pp**
- improved days: **1 / 5**
- bootstrap 95% interval: **[-0.24666pp, +0.12636pp]**

Request216 is slightly less bad than Request212's dynamic-minus-fixed day-balanced difference (-0.08459pp), but still worse than the fixed comparator.

## Interpretation

The exact-minute clock was not the primary bottleneck.

The remaining failure is more fundamental:

1. The current raw/aggregated state representation does not reliably distinguish continuation from failure.
2. One-step expected-return regression is a weak objective for a trader-like policy because the economically relevant question is not merely "is the next observed step positive?"
3. The same state can have noisy next-step returns while still differ meaningfully in pullback phase, recovery quality, and relative opportunity versus other candidates.

## Next research boundary

Do **not** lower entry/HOLD thresholds and do **not** add another minor variant of the same expected-return regressor.

Request217 should test a new representation/objective on the same already-opened block:

- explicitly encode pullback phase and turn quality from causal price/volume/transaction history;
- compare opportunities cross-sectionally within the market/time context rather than only against an unconditional zero-return boundary;
- use a path/risk-aware continuation target rather than a single next-step return;
- keep all evaluation thresholds preregistered and open no new holdout;
- if the new representation cannot materially improve ranking quality, retire this feature family before spending more holdout budget.
