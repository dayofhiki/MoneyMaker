# R298 findings — execution integrity restored, HOLD signal still fails

Source: run 36801235949, source commit 7fdd135584bef263f53f371f285785920712d772,
artifact 11136176759. May5-8 reused development only; June HOLD remains sealed.
97 candidate episodes, identical 86 entries, 37,627 states.

## Integrity

All six integrity checks passed. R297 reproduced exactly. All 86 positions in
all policies reconciled to regular-session next-print reference fills. Fixed
clock orders now submit on time. No delayed position was silently discarded.
Both 60s/300s label coverage increased to 100%, from about 60%/57% in R297.
This validates bookkeeping under the next-print reference scenario, NOT actual
broker execution, historical quote liquidity or live profitability.

## Returns on identical 86 entries

BASE friction-adjusted event means, not compounded account returns:

| Policy | BASE mean |
| --- | ---: |
| Immediate liquidation | -1.5004% |
| Fixed 60s | -1.4571% |
| Fixed 300s | -0.6722% |
| Fixed 600s | -0.8185% |
| Old 10m / 2% trailing | -1.6880% |
| Original 300s scores, corrected execution | -1.3144% |
| Rebuilt-label / refit 60s policy | -1.3384% |
| Rebuilt-label / refit 300s, primary | -1.4768% |
| Terminal timer plus common hard stop | -0.1074% |

Primary candidate cash-adjusted mean is -1.3093% over 97 episodes including 11
cash skips. This is an episode-weighted quantity, not a portfolio return.
Primary LIGHT mean is -0.6414%, STRESS -3.7113%; all four daily BASE means negative.
Refitting 300s labels worsens corrected original scores by -0.1624pp.

Primary gain versus old trailing: +0.2112pp across the SAME 86 entries, with
95% day-cluster CI [-0.1332, +0.8162]pp and ticker-day CI [-0.1730, +0.6182]pp.
This is not a convincing positive improvement, and neither policy is profitable.
Primary improvement versus immediate is only +0.0236pp.

OOF 60s target ranking Spearman is -0.0072; 300s is -0.0735. Even on labels with
both immediate and future fill delay <=3s, correlations are -0.0176 and -0.0992.
Missing labels alone were therefore not the cause of the weak signal.

## Opportunity and concentration

All 86 observed-window paths can now contribute to opportunity diagnostics:
28 reached +5%, 12 +10%, 4 +20% in observed opens before the cap.
These are observed-window excursions, not automatically tradable profits.
The refit primary realizes net +5% on only 1/28 of those +5 paths and net +10%
on 0/12 of the +10 paths. It fails to monetize much of the observed upside.

STFS still yields +75.50% in fixed 300s, but the record is now honest:
order submitted at +300s, reference fill at +3372s, delay 3072s (51.2 minutes).
That is an unusually long pending reference-fill outcome, not a liquid 5m win.
Removing STFS changes fixed 300s from -0.6722% to -1.5683%; terminal strategy
from -0.1074% to -1.3455%. No fixed-duration strategy is justified by this sample.

## Newly isolated training inconsistency

65/86 episodes hit the existing -2.5% modeled-net hard stop somewhere in the
observed path. Of the 37,627 training states, 26,943 (71.61%) occur AFTER the
first mandatory stop signal. A position following our rules cannot still be
held at those states. They were nevertheless included in R297/R298 HOLD fits.

Further, fixed-horizon liquidation labels can include a rebound after an
intervening mandatory stop. Thus the model is trained to value a continuation
that its execution policy is not allowed to take. Fixing timer bookkeeping did
not correct this action/target mismatch.

Nine entries breach the stop from unchanged-price BASE costs alone. Keep this
visible, but do not widen the stop or drop those positions to improve R299.

## Retrospective diagnostic, NOT an implementable strategy

For each existing position, choose the best saved next-open liquidation value
among decision states no later than its first mandatory stop (or observation
cap when no stop). This uses future knowledge and is only a favorable timing
ceiling under current entry/risk/reference-execution assumptions.

- mean +1.8984%, median -0.6689%; 33/86 could produce positive BASE liquidation;
- mean excluding the largest observation +0.6839%; this is still clairvoyant;
- among 28 observed +5 paths, 12 could realize net +5 before mandatory stopping;
- among 12 observed +10 paths, 4 could realize net +10;
- among 4 observed +20 paths, 1 could realize net +20.

There is some retrospective room to improve exits, but most entries still lack
a positive outcome under the risk constraint. This does not prove learnability
or an edge; it separates inaccessible rebounds from possible pre-stop exits.

## Decision

R298's integrity gate passed; its economic gate failed. Proceed to R299's
reachable-state and risk-consistent continuation ablation. Do not retune entry,
relax risk, choose a fixed duration from these means, or open June HOLD data.
