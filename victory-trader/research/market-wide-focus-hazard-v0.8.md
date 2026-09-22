# Market-wide learned focus hazard v0.8 — pre-registration

## Status

Frozen after request 124 completed and before outcomes from 2026-02-25,
2026-02-26, 2026-02-27, 2026-03-02 or 2026-03-03 are inspected. The failed
request-124 dates are not reused for tuning. April 2026 and later remain sealed.

## Question

Can a causal learned next-minute crossing ranker select a better 60-name focus
pool from the full eligible market than the frozen 0.60/0.40 threshold and
hysteresis policy?

Request 124 already showed that the downstream top-20 and HOT-10 layers retain
nearly all focus captures. This probe therefore changes only market-wide focus
admission. It is an attention-layer diagnostic, not a profitability claim or a
full hierarchy promotion.

## Frozen design

- Build one row for every eligible market-wide ticker at every completed
  minute while it has not yet crossed +10%.
- Label one only when the ticker first crosses +10% on the exact next completed
  minute; gaps without an exact next minute are ineligible.
- Fit only through 2026-01-16.
- Reuse the frozen v0.5 minute feature set and HistGradientBoostingClassifier
  hyperparameters; add no feature selected from request-124 outcomes.
- At each evaluation timestamp, rank all eligible rows by predicted hazard,
  break ties by ticker, admit at most 60 to focus, and retain the top 20 as the
  diagnostic shortlist.
- Compare against the unchanged threshold/hysteresis focus policy on the same
  five fresh sessions.

## Metrics

Report pooled and by session:

- first +10% runner crossings and exact-prior-minute focus capture;
- learned versus baseline focus capture;
- exact-prior-minute top-20 capture conditional on learned focus;
- average precision of learned hazard probability and raw attention score over
  all eligible market-wide rows;
- maximum focus and shortlist occupancy.

## Promotion gate

The learned focus admission hypothesis passes only if all are true:

1. every evaluation session has at least 10 first +10% crossings;
2. pooled learned focus capture exceeds the unchanged baseline;
3. pooled learned focus capture is at least 50%;
4. learned capture is non-lower than baseline on at least four of five days;
5. learned hazard average precision exceeds raw attention-score average
   precision;
6. top-20 capture conditional on learned focus is at least 95%;
7. focus occupancy never exceeds 60 and shortlist occupancy never exceeds 20.

If the gate fails, do not tune on these five dates. Use the pooled and daily
decomposition to decide whether the next untouched block should test temporal
persistence, capacity allocation, or a different causal label. Passing this
probe permits integration of the learned focus selector into the full frozen
top-20 -> one-second -> HOT-10 hierarchy on another untouched block.

## Result — request 125

Request 125 completed successfully on all five frozen sessions and passed every
promotion condition. Across 904 first +10% crossings, the learned focus-60 held
530 runners at the exact prior minute (58.63%) versus 304 (33.63%) for the
unchanged threshold/hysteresis baseline, a gain of 25.00 percentage points.

Learned versus baseline capture by day was 58.53% versus 24.81%, 60.12% versus
43.45%, 60.16% versus 47.15%, 59.47% versus 22.91%, and 53.91% versus 44.53%.
Thus the learned selector improved on all five days and every session exceeded
the minimum support requirement.

The model's market-wide average precision was 0.1469 versus 0.0681 for raw
attention score. The diagnostic top-20 retained 511 of the 530 focus captures
(96.42%), while focus and shortlist occupancy stayed exactly within 60 and 20.
The fit used 3,721,491 causal rows through 2026-01-16 and evaluation used
1,726,206 rows; no request-124 date was used for tuning.

This result promotes the learned focus selector only to a full hierarchy
integration test. It does not yet establish that the downstream one-second
reranker and HOT-10 allocation preserve the additional focus captures.
