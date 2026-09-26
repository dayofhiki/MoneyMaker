# Request244 findings — one-second execution reconstruction FAIL

Authoritative run: `36260935213`  
Artifact: `10912865151`  
Repository validation run: `36260935215` — full tests and Flat Files access passed.

No new market dates were opened. Request243 policy logic was not tuned from this result.

## Verdict

Request244 **fails its structural execution gate**.

Replacing sparse minute next-open references with a stricter observed one-second aggregate-open contract does not resolve Request243's missing execution. Under the primary 1-second latency and five-second expiry:

- archived episodes: **433**
- policy entries in this refit: **172**
- Request243-style unresolved entries in this refit: **72**
- entry fills observed within the contract: **65 / 172 = 37.79%**
- fully closed reconstructed trades: **30 / 172 = 17.44%**
- entry unavailable: **107**
- exit unavailable after an observed entry: **35**
- fully closed BASE mean: **-0.97952%**
- fully closed STRESS mean: **-3.00110%**
- positive closed trades: **33.33%**
- BASE losses <= -2%: **50.0%**
- all eight daily aggregate results remain unresolved.

The structural gate fails all four checks: entry coverage, exit coverage, total resolution and resolution uplift.

## Latency sensitivity

| Latency | Closed | Resolution | BASE mean | STRESS mean | Positive |
|---:|---:|---:|---:|---:|---:|
| 0 s | 34 | 19.77% | -0.80478% | -2.77223% | 32.35% |
| 1 s | 30 | 17.44% | -0.97952% | -3.00110% | 33.33% |
| 2 s | 29 | 16.86% | -1.19035% | -3.20027% | 31.03% |
| 5 s | 25 | 14.53% | -1.09308% | -3.03833% | 32.00% |

Coverage falls as latency grows, consistent with sparse observed prints.

## Important interpretation

Historical one-second aggregates are transaction-derived bars. A missing bar is **not** evidence that a market order could not have filled against a quote. Historical tick trades and NBBO/quote data are unavailable under the current Massive entitlement, so one-second aggregates cannot fully identify broker execution.

The experiment therefore does **not** establish that 62% of policy entries would literally be unfillable. It establishes that the currently available historical data cannot causally reconstruct those fills under a strict five-second observed-print rule.

There is nevertheless a useful alignment check. At zero latency, 33 trades were resolved both by Request243's minute next-open accounting and Request244's one-second reconstruction. Their mean BASE-return difference is approximately **-0.000004 percentage points**, effectively identical. This strongly supports timestamp alignment and the Request243 next-open price convention on cases where a same-time second print is actually observable.

At one-second latency, the paired mean difference is only **+0.04219 percentage points**. The execution proxy is therefore not hiding a large positive edge in the observable subset.

Most importantly, the observable closed subset remains economically negative. Missing execution alone cannot explain Request243's poor controller performance.

## Reproducibility note

The authoritative Request243 run reported 173 entries / 73 unresolved. Re-fitting the same chronological policy inside Request244 produced 172 / 72. The Request244 base differs from the evaluated Request243 source only by the recorded findings/results and a serialization-only `joblib.dump` change. No policy rule, feature, target, threshold or training date was intentionally changed.

This one-episode discrepancy is retained as a reproducibility warning rather than reconciled by tuning. It is too small to change the Request244 conclusion, but future policy experiments should prefer frozen serialized predictions/policies or hash-checked scored rows rather than silently assuming exact refit identity.

## Next research boundary

The next bottleneck is Request243's **action-value optimism / weak teacher**, not another execution threshold.

Request245 should remain diagnostic on the same May calibration block and quantify:

1. predicted ENTER value vs realized same-policy ENTER continuation;
2. predicted WAIT value vs realized same-policy WAIT continuation;
3. predicted HOLD advantage vs realized HOLD-vs-EXIT advantage;
4. direct predicted ENTER-minus-WAIT advantage vs realized advantage where both are identified;
5. sign precision and ranking quality of each value head;
6. target-support scale mismatch, especially the very compressed student WAIT target;
7. whether chosen ENTER and HOLD actions actually have positive realized advantage over their alternatives.

No thresholds, offsets, dates or policy actions should be tuned from Request245.
