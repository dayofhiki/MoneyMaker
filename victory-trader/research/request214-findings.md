# Request 214: achievable entry value and repeated observation

## Decision

The new entry controller was implemented and evaluated. **Do not promote it.**
It never predicts a positive executable 3-minute BASE entry value on the opened
development block. Zero trading is not evidence of a profitable trader.
No new market dates were opened, and no evaluation-driven threshold adjustment
was made. The controller is an entry diagnostic, not a completed recurrent
entry/exit system or a capital-path backtest.

## Verified recent trajectory

The authoritative comparison is the Request213 artifact from run
[36156827652](https://github.com/dayofhiki/MoneyMaker/actions/runs/36156827652),
artifact 10873848192. All figures below are per completed, evaluable trade,
not portfolio returns. BASE includes the repository's assumed execution costs.

| Policy | Resolved trades | Mean BASE return | Equal-day mean BASE return |
|---|---:|---:|---:|
| Request208 transition timing | 169 | -1.188591% | -1.114258% |
| Request209 state admission + same timing | 90 | -1.241862% | -1.175826% |
| Request213 supply admission + same timing | 85 | -1.457009% | -1.397369% |
| Request214 executable-value recurrent entry | 0 | undefined | undefined |

Recent filters reduced participation without improving the retained trades.
Request208 is a useful timing comparator, not a validated profitable policy.
Request209/213 train admission against the best realized entry over the future
watch window. That label measures an opportunity ceiling, not what the timing
policy can actually earn. Separately, Request208 forces a purchase at minute 5.
These are semantic mismatches to a trader who can wait and ultimately abstain;
they are not, by themselves, proof of feature leakage.

## Implemented change

- Reuse the existing hurdle model, model capacity, and transition feature family.
- Train against the return from ENTER at the current checkpoint, after BASE
  costs, rather than the best future entry outcome.
- At each observed minute, ENTER only if predicted net value is positive and
  ENTER is no worse than WAIT by the Request208 relative-timing model.
- Continue WATCH when conditions fail; at minute 5 still require positive value.
- Keep the 3-minute exit frozen to isolate this entry intervention. No claim that
  this is the final holding-period design.
- Separate the decision function from all realized outcomes. Preserve missing
  checkpoints and unresolved entry outcomes instead of counting them as cash.
- Include a one-shot comparator using the identical value model at minute 1.

## Results and what they mean

The five development dates are June 23, 24, 25, 26 and 29, 2026. Training is
April 30-May 8, calibration May 11-20. Inputs are the original Request171 fit
and calibration artifacts and Request178 development shards; their SHA256
hashes are recorded in `results/request214.json`.

There are 223 source episodes and 1,016 observed watch states. Predicted entry
values range from -3.402849% to -0.452807%; none exceeds zero. The repeated
controller records 794 waits, 168 fully observed abstentions and 55 episodes
whose observation sequence is incomplete. The one-shot controller abstains in
all 223 episodes. Entry-value rank correlation is only +0.0410.

A descriptive hindsight audit finds 152 positive realized labels among 814
evaluable watch-state labels, spanning 78 of the 223 episodes. These overlapping
labels are not 152 independent trades and cannot be selected using hindsight.
They show that the zero-entry result is a failure to identify profitable states,
not evidence that the sample contains no profitable price paths.

The reproduced Request208 decisions expose an accounting limitation: 201 entry
attempts include 32 whose 3-minute payoff cannot be evaluated; another 22
episodes hit missing checkpoints before an entry. The published 169 trades
are the remaining evaluable subset. Thus 54/223 (24.2%) episodes are not a
complete, economically resolved trajectory. They were counted in legacy
coverage statistics, but filling their returns with zero in an all-opportunity
comparison would conflate missing data with deliberate abstention. Request214
leaves this aggregate undefined and blocks advancement.

On the 169 evaluable Request208 trades, mean gross return is +0.013816%, BASE
return -1.188591%, and STRESS return -3.180701%. This describes the modeled
friction sensitivity of this sample; it is not a measured real brokerage fee.
The selected price moves are near flat on average before the modeled costs.
Reducing the decision threshold would manufacture more trades, not establish
positive expected value.

## Reproduction and validation

The initially available local numerical stack (scikit-learn 1.8.0, pandas
2.2.3, numpy 2.3.5) did not exactly reproduce Request208. After matching the
numerical versions printed in Request213's job log, all its daily and overall
mean returns were reproduced. The pinned stack is in
`request214-requirements.txt`. This is a reproducibility correction, not a
research-policy change. The canonical results above are the successful remote replay
[36162158697](https://github.com/dayofhiki/MoneyMaker/actions/runs/36162158697),
source commit `4eb38003c8ec09d1b1a3265e30556d0433518933`, artifact 10876506644.
The pinned local run is retained in `results/request214-local.json`. Python
locally is 3.12.14, remote 3.11.16. Predicted EV ranges and correlations differ
slightly between runtimes (local correlation +0.0400, remote +0.0410); we do not
claim bitwise model reproducibility. Both make exactly zero entries and
reproduce the Request208 comparator returns.

Validation: 636 full-suite tests passed under the installed local stack;
11 focused tests also passed under the pinned numerical stack. The new tests
cover waiting and re-entry decisions, terminal abstention, future-label
independence, missing outcomes, duplicate checkpoints and rejection of
zero-trade/negative/unresolved results. Critical Ruff checks passed. Remote full-suite CI and the experiment
workflow both completed successfully; workflow success is execution success,
not a passed research/profitability gate.

## Next research boundary

The bottleneck is identifying an achievable price move large enough to exceed
costs, not merely producing more buy/sell decisions. Do not promote either the
old forced-entry policy or the new all-cash policy. Before opening another
holdout, the next independent branch should build a complete event-time WATCH
and POSITION replay: preserve silent observations as explicit stale/no-trade
states, handle execution at actually available subsequent observations, and
learn holding decisions on trajectories produced by its actual entry policy.
Changing the holding period is a separate hypothesis; selecting the best
future horizon is not an executable policy. It must outperform cash after
costs on untouched periods and later pass portfolio/overlap/capital accounting.
