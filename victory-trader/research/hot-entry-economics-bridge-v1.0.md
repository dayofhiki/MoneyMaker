# First-HOT entry economics bridge v1.0 — conditional pre-registration

## Status

Prepared while request 138 is still running. This diagnostic may execute only
if request 138 passes its full attention/observation promotion gate.

It reuses exactly the request-138 evaluation sessions:
2026-05-07, 2026-05-08, 2026-05-11, 2026-05-12 and 2026-05-13.

It opens no later date. May 14 and later remain sealed.

## Question

Now that the system can find and maintain HOT candidates, do first-HOT episodes
actually contain enough post-signal movement to support a realistic entry
policy after modeled trading friction?

This is an economic bridge, not an executable entry strategy.

## Frozen episode definition

- one episode = one ticker-day;
- use the first chronological HOT state in that ticker-day;
- entry reference = the next exact minute bar open after the HOT decision;
- missing/non-positive entry reference is unevaluable and never shifted;
- no second HOT or later state in the same ticker-day may create another entry.

The next-minute open is a causal historical reference. It is not a claim about
a live fill.

## Frozen fixed-horizon diagnostics

For every valid first-HOT entry reference, report gross and LIGHT/BASE/STRESS
round-trip returns using exact future opens at:

- 1 minute;
- 2 minutes;
- 5 minutes;
- 10 minutes;
- 15 minutes;
- 30 minutes.

Use the existing execution-cost scenarios unchanged.

No fixed horizon is selected as a policy from these outcomes.

## Frozen hindsight opportunity ceiling

For each valid entry, inspect every exact observed future open from minute 1
through minute 30 and record the highest BASE round-trip return and its minute.

This is explicitly non-executable. It only answers whether enough economic
movement exists after first HOT for better entry/exit learning to be worthwhile.

Report:

- first-HOT episodes and valid-entry coverage;
- fixed-horizon gross/BASE/STRESS means;
- fixed-horizon BASE median, positive rate and p05;
- 30-minute oracle best BASE mean/median/positive rate/p05;
- median oracle-best minute;
- the same oracle diagnostics by trading day.

## Interpretation rule

This diagnostic does not promote a trading policy.

If the 30-minute oracle opportunity ceiling is weak or broadly negative, the
current HOT definition still does not provide an economically useful entry
population and ENTRY/ABSTAIN research should first improve candidate value.

If the oracle ceiling is clearly positive while fixed-horizon policies remain
weak, the primary problem is selective entry and adaptive exit timing. The next
research branch should learn:

1. whether this HOT episode is worth trading at all;
2. conditional on an eligible episode, when to enter;
3. after entry, repeatedly HOLD or EXIT using evolving state.

The already-known historical lessons from v3.3-v4.2 may be reused as design
evidence, but request-139 outcomes may not be used to claim an executable
profit edge.

May 14 and later remain sealed for a separately preregistered ENTRY/ABSTAIN
policy.

## Result — request 139

Authoritative successful run: `35827976308`. The probe reused the promoted
request-138 HOT trace and opened no date after 2026-05-13.

Across 2,091 first-HOT ticker-day episodes, the next executable minute-open
reference was available for 100% of episodes. Fixed holding times were not
economically viable under the frozen BASE friction model:

| Hold | Gross mean | BASE mean | BASE-positive rate |
|---|---:|---:|---:|
| 1m | +0.106% | -1.353% | 10.85% |
| 2m | +0.072% | -1.391% | 12.25% |
| 5m | +0.165% | -1.305% | 16.78% |
| 10m | +0.262% | -1.200% | 21.00% |
| 15m | +0.176% | -1.256% | 23.56% |
| 30m | +0.277% | -1.169% | 26.37% |

Thus HOT membership by itself is not an entry rule. The gross drift after first
HOT is only mildly positive and is overwhelmed by the frozen cheap-stock
spread/slippage assumptions.

The non-executable 30-minute best-exit ceiling was materially different:

- 2,042 / 2,091 episodes (97.66%) had at least one evaluable future exit;
- mean best BASE return was +0.339%;
- median best BASE return was -0.431%;
- 40.70% of evaluable episodes had some BASE-positive exit opportunity;
- median best exit minute was 10 minutes;
- per-day oracle mean was positive on all five sessions: +0.277%, +0.179%,
  +0.386%, +0.353%, and +0.505%.

The median remains negative and only about 41% of episodes have a cost-positive
exit even with hindsight. Therefore the result does **not** establish a
profitable policy. It establishes a sharper bottleneck: the validated HOT pool
contains real economic opportunities every day, but the majority of HOT
episodes should be skipped and fixed holding periods cannot extract the
available value.

### Decision

Proceed to a fresh learnability test for HOT economic opportunity. The next
branch must use only causal first-HOT state information to predict whether a
BASE-positive exit opportunity exists and to rank opportunity magnitude. It
must not tune a fixed holding horizon from request 139. May 14 and later remain
sealed until that experiment is preregistered.


## Result — request 139

Authoritative optimized workflow run: `35827976308`.

The diagnostic reused only the request-138 sessions and opened no later date.

Across 2,091 first-HOT ticker-day episodes, the exact next-minute entry
reference existed for 100% of episodes. Fixed holding rules were economically
weak under the frozen BASE execution model:

- 1m BASE mean: -1.3531%;
- 2m: -1.3912%;
- 5m: -1.3047%;
- 10m: -1.2000%;
- 15m: -1.2561%;
- 30m: -1.1690%.

Gross means were slightly positive at every frozen horizon, but were too small
to cover the current cheap-stock execution assumptions.

The non-executable 30-minute best-exit ceiling was materially better:

- 2,042 / 2,091 episodes evaluable (97.66%);
- mean best BASE return: +0.3391%;
- median best BASE return: -0.4309%;
- BASE-positive opportunity rate: 40.70%;
- median hindsight-best exit minute: 10.

The mean hindsight ceiling was positive on each of the five reused sessions,
from +0.1787% to +0.5049%, while the positive-opportunity rate ranged from
37.81% to 43.36%.

### Interpretation

Broad first-HOT entry is rejected. The validated HOT population does not itself
constitute an entry edge after modeled friction.

However, the positive mean hindsight ceiling on every session and the roughly
41% cost-positive opportunity rate show that the HOT population contains a
meaningful profitable tail. The dominant next problem is selective
ENTRY/ABSTAIN plus adaptive exit, not another fixed holding horizon.

Because request 139 collapses each ticker-day to its first HOT state, request
140 is opened as a same-date structural diagnostic of every actual HOT
promotion/re-promotion. May 14 and later remain sealed.
