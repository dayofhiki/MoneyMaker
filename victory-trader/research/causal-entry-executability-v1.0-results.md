# Request 165 — causal entry executability gate results

Authoritative fast-evaluation run: `36026026981`  
Artifact: `10819388739`

Request 165 used the frozen Request-160/161 attention architecture and
Request-163 economic-opportunity selector, then added one separate causal model
for exact-next-minute entry observability.

## Fresh block

- 2026-06-15
- 2026-06-16
- 2026-06-17
- 2026-06-18
- 2026-06-22

No model, threshold, feature or date was changed after fresh outcomes were
opened. Later execution changes were infrastructure-only sharding/vectorization.

## Calibration

The executability classifier threshold was selected only on already-opened
calibration history.

- threshold: 0.7702754917113916
- calibration selected rows: 1,453
- calibration exact-entry rate: 90.0206%

## Fresh result

Executability ROC AUC: **0.83248**.

The frozen economic selector alone produced 515 candidate first-HOT episodes:

- exact-next-minute entry rate: **71.65%**
- economic-label coverage: **70.29%**
- labeled oracle-positive rate: **57.18%**
- labeled oracle mean: **+1.3546%**

Adding the executability gate split those 515 candidates into:

- BUY: **280**
- WAIT: **235**

The BUY subset achieved:

- exact-next-minute entry rate: **91.79%**
- economic-label coverage: **90.71%**
- labeled oracle-positive rate: **56.69%**
- labeled oracle mean: **+1.4709%**

Thus entry observability improved by **+20.14 percentage points** while
oracle-positive rate changed by only **-0.49pp** and oracle mean actually rose by
**+0.1163pp**. The frozen economic non-regression gate passed.

## Fresh daily BUY support / entry rate

| day | BUY rows | entry-reference rate |
|---|---:|---:|
| 2026-06-15 | 64 | 89.06% |
| 2026-06-16 | 55 | 90.91% |
| 2026-06-17 | 52 | 96.15% |
| 2026-06-18 | 60 | 90.00% |
| 2026-06-22 | 49 | 93.88% |

Every day exceeded the frozen minimum of 25 BUY decisions and the frozen 85%
daily entry-observability floor.

## Mechanical fixed-horizon diagnostics

The BUY subset still did not show positive mean BASE-net return under fixed
holding periods:

| horizon | BUY mean BASE net |
|---:|---:|
| 1m | -1.202% |
| 2m | -1.188% |
| 5m | -0.898% |
| 10m | -0.970% |
| 15m | -0.888% |
| 30m | -1.342% |

These horizons are diagnostics only. None is promoted or selected from the fresh
block.

## Formal decision

**PASS**.

All preregistered Request-165 entry-bridge conditions passed:

1. >=25 BUY decisions every day;
2. pooled BUY exact-entry rate >=90%;
3. daily BUY exact-entry rate >=85% every day;
4. fresh executability AUC >=0.55;
5. oracle-positive-rate non-regression;
6. oracle-mean non-regression.

## Research consequence

The primary missing-entry problem from Request 164 is now substantially solved
by a causal BUY/WAIT split. This validates executability as an explicit entry
state rather than a bookkeeping afterthought.

The remaining core bottleneck is still realized trading economics. A fixed
holding time remains negative after BASE costs. The next branch should therefore
keep the entry bridge frozen and build a genuinely recurrent policy:

HOT -> BUY / WAIT / ABSTAIN, repeatedly while the opportunity remains live;

POSITION -> HOLD / EXIT, repeatedly from causal post-entry state.

The next development objective is not another fixed horizon. It is to learn
state-dependent entry timing and state-dependent exit timing, then test the full
capital path on a later unopened block.
