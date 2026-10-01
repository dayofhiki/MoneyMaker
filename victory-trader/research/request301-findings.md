# R301 findings — history availability fixed, economic gate still fails

Official run: https://github.com/dayofhiki/MoneyMaker/actions/runs/36831962753
Source ba734710eda099fe8c6b51eb06ed97cb503182b3; artifact 11147441032.
Execution succeeded; economic gate FAILED. Official JSON equals complete local
verification. R300 C exactly reproduced; same 86 entries all resolved. May5-8
development only, June HOLD sealed. No parameter changes following results.

## Outcomes

| Measure | A: R300 C | B: R301 primary |
|---|---:|---:|
| Mean BASE net % | -1.473654 | -1.468004 |
| Median BASE net % | -1.227184 | -1.320558 |
| Positive entries | 5/86 | 8/86 |
| First-decision model exits | 56 | 44 |
| Median holding seconds | 9 | 14 |
| Hard stops | 14 | 18 |
| CVaR5 % | -5.251287 | -5.319604 |
| Mean gross % | +0.069138 | +0.075017 |

B-A +0.005650pp, day CI [-0.023946,+0.058061], ticker-day
[-0.191743,+0.200731]. Effect near zero and uncertain. B-R299 D -0.121237pp,
day CI [-0.220713,-0.058228], ticker-day [-0.307684,+0.039943].
B daily net means all negative: -1.951912, -1.134185, -1.931310, -1.149652%.
LIGHT -0.632498%, STRESS -3.702968%, largest-winner-excluded -1.542791%.
Mean gross-to-BASE cost drag 1.543021pp. Sparse next-print references: 46/86
exits wait >3s. Nine cost-only stop breaches retained. These are position means,
not portfolio returns or proof of executable quotes.

## What the feature change accomplished

At 75 first reachable states, all price-return and 60s close-range features are
available in B versus none in A; volume ratio available 58/75 (zero/absent prior
window volume remains NaN). All 68 B model-exit states have 60s price context
versus only 3/72 A model exits. Thus the specific post-entry-history availability
problem is resolved under the saved regular-session data contract.

Behavior changed, but later exit is not uniformly better. Realized outcomes
change for 28 positions: 13 improve, 15 deteriorate.

| Ticker/day | A BASE net % | B BASE net % | A hold s | B hold s |
|---|---:|---:|---:|---:|
| BGDE May6 | -0.429282 | +3.719498 | 1 | 11 |
| DGXX May6 | -1.169740 | +1.475688 | 1 | 20 |
| ZJK May6 | +4.177989 | -0.650206 | 39 | 10 |
| EOSE May8 | -0.922969 | -2.580257 | 1 | 103 |
| VTIX May8 | -1.229634 | -2.871896 | 166 | 2157 |

BGDE/DGXX support the value of preserved context in individual cases. ZJK and
EOSE/VTIX show that neither a longer nor shorter generic holding rule follows
from these examples. Quotes/fill gaps also qualify the apparent path economics.

## Learning diagnostics and interpretation

Own-target weighted rank A +0.056993 -> B -0.028450; first-decision rank
A +0.167739 -> B +0.046109. B's aggregate unweighted Spearman +0.107557 is
not a substitute for the preregistered weighted metric. Own pi1 labels differ
across feature policies, so common labels provide a distinct diagnostic.

On common A-pi1 FIRST-decision targets, B's weighted sign accuracy 59.375%
versus A 57.619%; B predicted-HOLD group (31 states) weighted realized advantage
+0.357675pp, predicted-EXIT group (44 states) -0.147674pp. On frozen-pi0 labels,
B predicted-HOLD group advantage is -0.004024pp and first-state weighted rank
-0.021575. Interpretation depends strongly on which downstream policy follows
WAIT. These are future realized counterfactual advantages, not profit/optimality.

Across A and B pi1 downstream targets, signs differ at 1,509/10,619 reachable
states (14.21%), mean absolute label difference 0.620979pp. At first decisions
only 3/75 change sign. Frozen pi0 versus A-pi1 target signs differ at 21/75 first
decisions and 4,776/10,619 overall states. Policy improvement is intended to
change downstream outcomes; these differences do NOT prove target corruption.
They motivate separating target-policy quality and stage-to-stage calibration.

## Research decision

No promotion. Preserve the causal history implementation for diagnosis, without
claiming it improves profitability. Stop expanding features on these four days.
Next preregister a fixed-reference / stage-by-stage continuation-policy audit:
compare outer-day pi1 and final pi2 using the same full-history feature contract,
separate common target quality from self-policy targets, and examine actual
first/visited HOLD decisions, sign calibration, risks and all-entry economics.
Do not tune a threshold, impose minimum holding, widen stops or select examples
after outcomes. Entry/cost economics remains an alternative bottleneck: gross
edge +0.075% is far below friction, but that does not establish that a better exit
cannot recover more gross return. Next experiment has not been launched.
