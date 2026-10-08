# R330: continuous action bridge implemented, but the ENTER value head chooses only cash

**8/18 checks pass,10 fail. F/E are not adopted.** A cost/stop/fill-aware continuous
ENTER/WAIT policy and shared capital replay now exist. They expose the next
bottleneck: sufficient independent economic action targets and a learnable
positive absolute ENTER value. This experiment does not establish beneficial
waiting, a profitable allocator or a deployable trader.

## Fixed design and policy outcomes

Preregistrationb1bc6c5b8fac2d14f52e7177753e9cdf98d6db73 precedes new labels/fits.
Use7,806 original May6–8 training observations and19,530 May11–20 observations,
all828 evaluation identities, strict-past/frozen G and instantaneous coordinates.
Two10-input linear ridge heads estimate immediate ENTER BASE payoff and WAIT
continuation under fixed causal pi0. No outcome enters inference. Chronological
May7 fits May6, May8 fits May6–7; May6 value predictions remain unsupported/NaN.
All original clocks independently replay from authenticated existing raw caches.

Downstream ENTER: next-print entry/exit<=3s; BASE−2.5% absorbing stop at completed
closes; fixed60s hold with original HOT/session cap; unchanged LIGHT/BASE/STRESS.
No unknown order is converted to cash and no missing deadline uses an earlier
price. WAIT labels follow the next action of fixed pi0, not a future maximum.
Q_WAIT is Q^pi0, not optimal WAIT or the learned F policy's own value.

| Policy | Episode entries | Resolved trades | Known cash/closed episodes | Full BASE mean | Known-positive opportunity capture |
|---|---:|---:|---:|---:|---:|
| F: Q_ENTER>max(0,Q_WAIT), continuous |0|0|828/828|0%|0/69|
| E: same Q_ENTER>0, continuous |0|0|828/828|0%|0/69|
| G: G>=prior and cost<2.5%, continuous |330|59|557/828|unknown|7/69|
| G0: G rule at first clock only |330|59|557/828|unknown|7/69|
| CASH |0|0|828/828|0%|0/69|

G/G0 select exactly the same episode entry indices here; continuous reassessment
under this simple threshold adds no entries. This does not show all recurrent
policies lack value. F/E also share every action, because **Q_ENTER is always
negative**: full-training range[−2.129167,−.611352]% and evaluation
[−2.648053,−.571570]%. The maximum chronological supported value is−.513230%.
Thus this run cannot isolate a useful or harmful WAIT comparison. Do not count
all-cash resolution100%, zero losses or zero exposure as aggressive trading skill.
There are69 episodes with an observed positive fixed-contract BASE outcome, but
that is a future diagnostic and a lower bound on observable opportunity, not an
oracle deployable policy or permission to select those episodes.

## Target support and economic learning failure

Common ENTER/WAIT targets:4,144 rows from104 independent episodes,34 positives
over all3 dates. This fails the registered>=150 support floor despite passing
the positive-episode floor. Original episode observation-weight mass266 becomes
34.640426 on common targets; censored rows are not renormalized. Training mean
ENTER−1.185723% and WAIT−.083819% on this common weighted support. Ridge normal
equations have condition235.697240 and maximum residual2.842171e-14: the cash
result is not a failed optimizer or fabricated unsupported fit.

The post-result descriptive audit counts ALL immediate finite ENTER targets:
4,304 rows/114 episodes,38 positives,original mass38.638358,weighted mean−1.132952%.
Removing WAIT-target availability alone would still leave114 episodes, so that
small loss change is not a sufficient support repair. Evaluation has10,655 finite
immediate outcomes/263 episodes; their original mass91.736289 of679 observed
episode mass illustrates how many dense rows obscure limited independent coverage.
Neither conditional mean is the complete candidate/account expectation.

The value head has descriptive positive-payoff AUROC.589801 on finite evaluation
targets, while all its predicted economic values are negative. That rank statistic
cannot promote it. This does not identify whether true expected edge is negative,
the linear mean model is too limited, the objective/representation mismatches the
fixed horizon, or censoring biases the supported sample. Resolve support and
execution observability before assuming a more complex model fixes economics.

## Execution and capital bottleneck

Of330 G/G0 episode orders,215 have late/missing entry references and56 have
late/missing exits.59 resolve with conditional trade BASE mean−1.181321%.
Including498 cash episodes, the557 known outcomes have mean−.125131%; the other
271 are unknown, so the full828-episode expectation remains unknown.

Paired conditional F−G mean intervals are positive: day[.086191,.152745] percentage
points, ticker-day[.077996,.170641]. They compare cash against observed negative
returns on only557 common-resolved identities. They are not evidence of positive
F trading, a full-population improvement or a fresh validation. F−E intervals
are exactly[0,0]. All failed full-population mean gates remain failed/unknown.

Separate account replay uses2 slots,10% initial-day notional and causal priority,
capacity WAIT/reassessment, same-ticker blocking and release only after an exit
fill. F/E/CASH remain cash with return0. G/G0 each attempt24 portfolio orders;
two unknown reservations occur on every date, leaving all8 daily account
returns and the aggregate unknown. Known-leg returns are explicitly partial.
Closure-ledger drawdown omits intratrade marks; full marked drawdown remains
unidentified. This shows why independently selecting attractive-looking tickers
is not enough for a shared-capital trader, without proving an allocation policy.

## Reproduction and operational deviation

Canonical first official [run37748181012](https://github.com/dayofhiki/MoneyMaker/actions/runs/37748181012),
source29b60b70ec02e68292f19c7d9c0af80017da1cc5/artifact11536504167, is authenticated.
Its909 JSON numeric fields have local replay error0; main JSON bytes and all5
parquet tables are exact. Identical labels, missingness, identities, Q/priority
tie-aware ranks and actions are verified. Atomic writes leave no temporary files.
Archive hashes/source checks are request330-provenance.json.

**Protocol deviation: an unnecessary second cached official replay occurred.**
Commit/run listings omitted the custom replay, and I changed the trigger before
checking commit check-runs, even though the first replay had succeeded. Retain
[run37748621105](https://github.com/dayofhiki/MoneyMaker/actions/runs/37748621105),
source0de204621a97052ba5dc39adf9f94af7009631ec/artifact11536603199. Scientific code,
inputs, coefficients' fitting rule, parameters and seeds are unchanged. Duplicate
JSON maximum error3.694822226e-12 and table error7.527312107e-14 satisfy the ORIGINAL
1e-9 tolerance; all ranks, actions, decisions and account outcomes are exact.
Three state parquet bytes differ numerically, while decision/order tables match
bytes. Do not claim duplicate bit equality or count it as another sample/fresh
validation. The earliest successful replay is canonical; no favorable result is
selected. Future execution checks must use commit check-runs before retriggering.

Local full tests1,220 pass (160 existing/expected warnings);15 new action/account
tests and critical Ruff pass. Implementation, canonical-source and duplicate-
source CI pass. One-shot workflow is retired at closeout. No new market requests,
June HOLD/July-August opening, event router, main merge, live trades or promotion.

## Next experiment

Keep memory low priority and retain the current-state action/ledger structure.
The next proposal is request331-proposal.md: a separately registered **action-
target observability census**, checking original early support and fixed later
5–60min observation phases from existing raw caches under the EXACT60s/−2.5%/3s
contract. Preserve all identities and phases; do not select profitable clocks,
relax fills or call additional seconds independent examples. This proposal is
designed, not yet registered/executed. If it does not supply adequate independent
economic labels, improve execution observations/data rather than invent memory
or hide unknown account returns. R316/R316B stays an independent specialist
candidate, with insufficient evidence for routing or profitable simple policy.
