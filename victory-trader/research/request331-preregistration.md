# R331: fixed cached action-target observability census

Register this file and request331-inputs.json remotely before constructing any
new clock, price feature or outcome. R330 is closed at
6c72824d36a5f86d4f282b0cb5f201d1d136eac7. Its fixed ENTER head chose only cash;
explicit memory remains low priority. This is an unfit census, not a policy,
holding-period search, profitable phase selection or validation dataset.

## Scope and clocks

Authenticate every byte in request331-inputs.json before reading tables. Existing
R315 broad training membership is exactly 407 identities on May5–8 (102/108/97/100).
Primary feasibility scope is the 305 May6–8 identities; report May5–8 separately.
Evaluation is exactly all 828 R319 identities on May11–20. Keep every identity,
including no-print and all-censored episodes. No acquisition, additional dates,
June HOLD, July–August or R316/R316B membership changes.

Use regular cached second timestamps only to form clocks, with completed time
`t+1000`. EARLY retains every original print whose start is at/after HOT, completion
is at/before HOT+300s and strictly before min(HOT+3600s, session close).
ADDITIONAL_LATE uses the first completed print in each fixed completion bucket
`(HOT+300s+30s*j, HOT+300s+30s*(j+1)]`, j=0..109, strictly before the same terminal.
Bucket membership and first choice use timestamps only, without price, volume,
future density or outcome. Empty buckets create no observation. EXTENDED is the
union, with identical early clocks. Publish both scope manifests and complete
identity clock ledgers before computing any price or outcome. Validate early
clock equality to R319, and prefix/future-price invariance in tests.

## Fixed economic contract and support definitions

Reuse R330 enter_outcome unchanged: first next regular second open at/after the
decision, entry/exit lag<=3s, completed-close BASE −2.5% absorbing stop, fixed60s
hold capped by original HOT+60min/session terminal, unchanged LIGHT/BASE/STRESS.
Missing or late orders remain NaN. No earlier-price deadline substitution, future
best exit or unknown-as-cash. Known BASE cost is `-net(last completed close, same
close)`; cost-compatible means finite known cost strictly <2.5%, available before
any order. Verify all inherited May6–8/evaluation early costs/outcomes against
canonical R330 within original 1e-9, with exact identities, missingness and reasons.
No G transport to late phases, no WAIT labels or learner are fitted in this census.
Current 120s features are deferred to a separately registered learner.

For EARLY, ADDITIONAL_LATE and EXTENDED, report observed rows/identities, resolved
immediate targets, independent episodes with >=1 resolved target, and positive
(>0)/nonpositive (<=0) BASE target episodes both unfiltered and cost-compatible.
Keep all unknown rows and report per-date support, fill-delay distributions,
stop/cost-only breaches and payoff distributions under all three cost scenarios.
Full population mean is unknown whenever any observed target is unknown; known
means/quantiles are explicitly conditional and never a feasible trading return.
Positive episode support is only an observability diagnostic, not an oracle policy.

Use original R330 row-weight rule 1/(number of ALL observations in that episode
and declared arm), before target/cost filtering. EARLY reproduces inherited weights.
EXTENDED recomputes this declared observational denominator over the union;
it does not renormalize available targets. Report observed episode mass, complete
and cost-compatible complete mass, and separate early/late contribution under the
EXTENDED denominator. More rows and reallocated weight are not independent samples.
Save a paired identity ledger with early/late/extended row counts, any-complete,
any-positive and any-nonpositive flags, plus gained/lost/unchanged support. Loss
of any early complete identity in the union is an integrity failure.

## Registered gates and uncertainty

Primary scope May6–8 EXTENDED: >=150 independent episodes with at least one
resolved cost-compatible ENTER target; >=30 episodes with a positive compatible
BASE target; positive and nonpositive compatible episodes both occur on all3
dates. Execution-conditioned row resolution (resolved / ALL causal cost-compatible
clocks, not post-fill-selected rows) must be >=90%. Empty denominator fails.
Report rates on all rows and at episode level as additional diagnostics.
No substitution of unconditional counts or evaluation profitability into gates.

Also require complete input authentication, exact populations, early preservation,
early R330 contract reproduction, clock-manifest-before-target publication and
causal cost/clock invariance tests (6 integrity checks; 10 total including the4
feasibility requirements). Preserve every failed requirement. Integrity failure
stops completion claims; underpowered support still gets the fixed census.

Day/ticker-day clustered 95% support intervals use1000 bootstrap multiplicities,
fixed seeds20266460/61, original equal-day/equal-identity weights including
unobserved identities. Report observed, any complete compatible and positive/
nonpositive compatible episode rates and paired EXTENDED−EARLY rate increments.
These are fixed-census descriptive intervals with only3 primary dates; no refits,
no new information and no claim of population validation. Clock resolution uses
within-episode original observation weights for the clustered denominator.

## Validation, execution and decision

Tests cover fixed bucket boundaries, empty buckets, all-identity retention,
timestamp prefix invariance, future/label independence of cost, inherited contract
replay, terminal/missing references, no complete-target reweighting and paired
support. Atomic parquet writes require readable complete footers. Run full tests
and critical Ruff. Execute one cached official replay, inspect source check-runs
before any retrigger, and preserve any failure or duplicate. Authenticate source/
artifact digest, compare JSON/table values within1e-9 and exact identities,
clock membership, missingness, support flags/reasons and payoff tie-aware ranks.
Retire the one-shot workflow after verified closeout.

If all feasibility gates pass, separately preregister current-state economic
learning on ALL declared phases, rebuilding chronological quality/execution/value
components. Do not select the best phase or sweep learner settings in this census.
If they fail, prioritize explicit execution-observation/source or justified order
contract design before another value/memory model. A next-print proxy remains a
scenario assumption. Learned HOLD/EXIT, capital opportunity cost and ADD/REDUCE
remain later questions. No main merge, promotion or live trades.
