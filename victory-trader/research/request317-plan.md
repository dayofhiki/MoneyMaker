# R317: broad continuous-observation training under frozen evaluation

Registered before constructing new states, labels or fits. Parent main
18e46637a8b237f4d2e6c6f2717bab5df7ae8ff5, after official R315. R316/316B
remain independent draft event tracks. May is reused development. June HOLD
and July-August remain sealed. No new market requests or policy promotion.

## Fixed input and observation contract

Reuse official R315 run37450538746:478 frozen union identities,407 broad
SHA256-selected HOT episodes and580,944 raw rows on May5-8. Verify original
raw/cohort hashes, unique identities and source dates. Read no new population.
Use only407 broad episodes for new fits; preserve all missingness in a ledger.

Create a cash-state decision at the end of EVERY completed regular one-second
aggregate with start>=HOT and end<=HOT+300s, and end<min(HOT+1h, session close).
300s is the inherited R306/R315 early observation limit, not a searched delay.
No forward-filled pseudo-seconds, winner-selected clocks, stop after good labels,
or stopping the hypothetical cash observation after an earlier hypothetical buy.
States are alternative cash decisions, not sequential executed positions.
Membership and clocks depend only on known bars, never later outcomes.

Freeze membership/clocks before labels; compute unchanged M28/Q9 clock features
from aggregates ending at/before each decision. HOT background context remains
the original R315 context on every state, matching the inherited feature contract.
Labels use next-open entry, unchanged BASE costs, -2.5% absorbing stop and
HOT+1h/session cap. Missing terminal/entry labels remain censored. Keep missing
episodes, first-censored/later-complete transitions and ever-positive counts.
Episode counts, not seconds, measure independent support.

## Fixed comparisons

- R: replay original M on4,526 R310 repeated selected states, seed20264605.
- B: replay broad first-only M on the original334 complete R315 first states,
  seed20265005; require the saved R315 B probabilities to match within1e-9.
- C: M28 on all complete continuous broad states, seed20265105.
- Q: Q9 on EXACTLY the same C rows, same seed; isolates directional increment.
- H: M28 continuous states restricted to the same334 complete-first episodes
  used by B. H vs B isolates schedule within matched episodes; C vs H describes
  additional episodes with censored first labels and later complete states.

All use unchanged fit_linear, C=.1, lbfgs, tol1e-8, max_iter2000,
training-only transforms, single-class prior fallback. independent_weights gives
equal day total, equal episode total within day, equal complete state weight
within episode, with total effective weight=independent episode count. No model,
feature, observation-window, seed, threshold or calibration search.

Score all679 observed original R311 evaluation cases and retain all828 identities.
Evaluate at original first clocks and SAME650 final labels; no evaluation-label
updates or later evaluation states. C-B is the total schedule/support change,
H-B the matched-episode schedule contrast, C-Q the directional increment, C-R
the original-model comparison. None is an identified market causal effect.

Chronological diagnostic: May6 fits May5, May7 fits May5-6, May8 fits May5-7.
Held cases are original broad canonical first states. B/C/Q/H transforms and
labels come only from preceding training days; May5 is unscored. Require
nonempty fitting support. Chronological diagnostic dates differ from evaluation.

## Reports and fixed decision

Primary day/episode-weighted same-day AUROC, pooled AUROC/AP/Brier, own-prior
Brier, per-day metrics and coverage. Fixed-score paired day and ticker-day
bootstrap1,000 draws with seeds20265150/20265151; report invalid draws for
C-B,H-B,C-Q,C-R. These intervals do not cover fitting uncertainty.

Fixed checks: evaluation coverage>=90%; >=20 positives on>=4 dates;
C same-day AUROC/AP/Brier beats B/R/Q and Brier beats own prior; C-B improves
AUROC on a majority of evaluable dates; ticker-day same-day CI lower>0 for
C-B,H-B,C-Q. Keep every failed check. Even all-pass cannot promote a policy.

Persist input hashes, outcome-independent clock manifest, all state/episode
missingness, fit/fold audits and scored evaluation checkpoint. Run synthetic
scope, causal-input, missingness, matched-row, weighting and chronological tests.
Publish branch-only official cache replay, reproduce outputs locally within1e-9
including identities, missingness, probability ranks and flags. Retire the
one-shot branch workflow after closeout. main/research-request.json stays intact.
