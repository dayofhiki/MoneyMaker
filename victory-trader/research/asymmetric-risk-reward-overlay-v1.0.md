# Request 194 — asymmetric risk/reward overlay bridge

## Status

Pre-registered after Request 193 failed its frozen shadow-entry observability bridge and before implementing or running Request 194.

Request 193 showed that delayed entry states still contain substantial hindsight 30-minute opportunity on average, but the current representation cannot rank which delayed state is a good new entry. Separately, Requests 187/189/191 showed that the current BUY population contains longer-horizon upside and that post-entry path information becomes informative.

Request 194 therefore isolates a different question before adding another predictor:

**Can a simple, causal, asymmetric position-management overlay materially change the economics of the existing BUY population by cutting losers and preserving the right tail?**

This request does not change candidate discovery or entry selection. It is a position-management development bridge.

## Data

Calibration / policy selection:
- Request-171 chronological calibration POSITION rows only.

Fresh development evaluation:
- Request-178 POSITION shards on Jun 23/24/25/26/29.

No new dates are opened. Request-178 remains development-only and is not promotion evidence.

## Causal execution semantics

At each completed one-minute POSITION state:
- compute marked BASE return from the original modeled entry fill to the completed current close;
- use that marked value only as a causal trigger;
- if a rule fires, execute at the already defined next causally executable open reference (exit_now_base_return_pct).

Therefore a -5% stop is **not** assumed to fill at exactly -5%. Gaps and modeled execution drag remain in the realized result.

If the required executable open is missing, the trajectory is unresolved rather than receiving an idealized fill.

The final research cap remains 30 minutes. At minute 29, the existing exact minute-30 reference is used when available.

## Frozen policy family

The family is fixed before fresh evaluation:

- hard-stop marked BASE thresholds: -3%, -5%, -7%;
- first partial-profit thresholds: +5%, +10%, +15%;
- trailing gaps on the remaining position: 3pp, 5pp, 7pp;
- partial fraction: exactly 50%;
- after the partial take, update the peak causal marked BASE return each minute;
- exit the remaining 50% when marked BASE falls by at least the trailing gap from that post-partial peak;
- hard stop has priority over profit-taking/trailing;
- if neither rule completes the trade, force the remainder out by 30 minutes.

This gives 27 predeclared policies. No threshold is added or removed after looking at fresh results.

A canonical human-style reference rule (-5% stop, +10% half take, 5pp trail) is reported whether or not calibration selects it.

## Calibration-only selection

Evaluate all 27 policies on Request-171 calibration rows.

Eligible policies must have >=90% completion coverage.

Select exactly one rule by:
1. highest day-balanced BASE mean return;
2. if tied, lower severe-loss rate (BASE <= -5%);
3. if still tied, shorter mean hold;
4. if still tied, lexicographic (stop, take, trail) order.

Fresh dates are never used for policy selection.

## Fresh diagnostics

For the selected policy report:
- completion coverage;
- mean and day-balanced BASE return;
- bootstrap 95% interval over daily means;
- positive-trade rate;
- mean winning return;
- mean losing return;
- payoff ratio = mean win / abs(mean loss);
- severe-loss rate;
- mean / median / p90 hold;
- partial-take frequency;
- hard-stop frequency;
- trailing-exit frequency;
- positive-return day count.

Comparators:
1. minute-1 exit;
2. hold-30;
3. same selected hard stop with no partial take/trailing;
4. same selected partial-take/trailing rule with no hard stop;
5. the canonical -5/+10/5 rule.

Matched per-episode differences versus minute-1 and hold-30 are required.

## Reference account

Use the frozen Request-192 ledger assumptions:
- start KRW 1,000,000;
- maximum 5 concurrent positions;
- each admitted position receives up to 20% of current nominal equity, bounded by available cash;
- process exits before entries at the same timestamp.

For partial exits, return the corresponding 50% principal plus its realized return to cash at the partial-exit time while the remaining half keeps the position slot until final exit.

Report:
- starting balance;
- ending balance;
- net P/L;
- total return;
- admitted trades;
- skipped-capacity count;
- completed-episode coverage;
- win/loss counts;
- realized ledger peak-to-trough drawdown (not intraminute mark-to-market DD).

## Frozen development gate

Pass only if all hold on Request-178 fresh development dates:

1. completion coverage >=90%;
2. selected policy day-balanced BASE mean >0;
3. daily-bootstrap 95% lower bound >0;
4. positive mean return on at least 4/5 fresh days;
5. matched day-balanced difference versus minute-1 >0 with bootstrap lower >0;
6. matched day-balanced difference versus hold-30 >0 with bootstrap lower >0;
7. severe-loss rate is lower than hold-30;
8. reference account ends above KRW 1,000,000.

A pass does not promote the model. It only licenses integrating the overlay with a future honest WAIT / ENTER / ABSTAIN entry policy and then validating on untouched dates.

## Interpretation discipline

This experiment answers whether asymmetric money management can rescue or materially improve the economics of the **current entries**. It does not prove that the entry edge is solved.

If the overlay improves tails but remains negative, the bottleneck remains new-entry economic timing/repricing.

If the overlay becomes robustly positive, risk/reward management becomes a first-class MoneyMaker component rather than a downstream safety patch.
