# R306 preregistration — cash-state momentum observability

Fixed before the first R306 research fit. R305 official run37210193519 passed
execution but did not establish early upside observability. Keep May5–8 only,
all97 saved R294 shortlisted episodes including11 abstentions, exact86 frozen
R296 entries as an unchanged reference, BASE/LIGHT/STRESS costs, absorbing
-2.5% BASE-net hard stop, and common hot+60min/session cap. No June access.

Data: R294 run36721315072 scored OOF states and R298 run36801235949 regular
raw seconds. R298 covers86 ticker-days. Fetch/cache only the11 missing original
May ticker-days, once each, using unadjusted second aggregates. No new dates,
new universe or duplicated acquisition for the86 covered pairs. Save the whole
raw input in the R306 artifact. Refuse to fit the primary research if coverage
is incomplete; a partial-input mode may build labels/features for engineering
verification only, with no model fitting or research conclusions.

Bottleneck hypotheses: (1) original fixed-horizon MFE+MAE entry utility omits
costs/absorbing stop and can reward rebound paths unavailable after liquidation;
(2) a last-N-print turn mixes quick sustained demand with sparse noisy prints;
(3) first-state signal differs from information available after cash rechecks.

At every ORIGINAL saved pre-entry state, map hypothetical submission to first
regular print open, then evaluate all completed observations after that fill.
Observe BASE-net stop from completed closes; stop submission is absorbing and
its pending next-print fill is retained. Allow terminal submission at the
common cap. Calculate future maximum net return on that allowed schedule,
cost-cover and >=5/10/20 labels plus stop-ignoring ceilings. These are future
outcome labels only, not selected best entries, optimal policies or returns.
Missing terminal fills are censored. Include cost-only immediate stops.

Features: baseline uses ONLY R294 current-second raw features, event features,
turn features and10 named static causal context columns. Exclude all saved
model predictions, setup scores, execution prices/times, future utility/MFE,
oracle labels, selection thresholds and learned upstream admission scores.
Frozen shortlist selection is still conditional upstream OOF, not nested.

PRIMARY C: add fixed clock features from strictly completed raw regular seconds
through decision_t-1s, retaining pre-HOT history. Fixed windows10/30/120s:
endpoint return, signed close-path efficiency, volume/transaction rate relative
to preceding same-length clock window, positive-volume proxy imbalance and
prior-window high reclaim. Also activity fractions and elapsed gap, BASE
roundtrip cost estimated from last completed close, and short/long acceleration.
Activity includes absent seconds, never zero-filling absent prices. Trade-side
direction is an aggregate close-change proxy, not buyer-initiated order flow.

Matched outer-day arms: A same baseline features, regression on historical
MFE+MAE utility; B same baseline features, classification of reachable net+5;
C baseline+clock features, classification of reachable net+5. Also B/C cost
cover, net+10 and net+20 classification for descriptive sparse-tail support.
D COST CONTROL: baseline plus only known cost drag, same target/models. This
control was specified before any R306 research fit (partial engineering builds
fit no models), to prevent attributing a low-price/cost filter to momentum.
Run D at all four target levels; require C also improves D AUROC/AP/Brier.
Identical HGB lr=.04,220 iterations,15 leaves,min-leaf75,l2=3,no early stopping,
seed20264100+fold*100. No balancing or threshold selection; day/episode weights.
Single-class training folds use training prevalence; no inversion of scores.

Report paired weighted AUROC/AP/Brier and training-prevalence baselines on all
states and fixed cash snapshots first,+20,+60,+120 seconds after each episode's
first saved decision, choosing first available state at/after each timestamp.
Show97-denominator coverage, matched-episode comparisons and per-day metrics.
Cluster bootstrap C-B and C-D on common first snapshots by whole day/ticker-day; do
not count correlated seconds as independent evidence. Primary net+5 signal
claim requires C>B AUROC/AP, improved Brier and positive daily AUROC difference
on at least3 evaluable days; all declarations remain exploratory development.
No policy deployment, fresh validation or forced waiting on a diagnostic gain.

Report exact frozen-entry mapping, original utility versus risk/cost ceiling,
opportunities lost to absorbing stop, causal feature availability and clock
gaps. Tests cover future-feature invariance, sparse timestamps, pre-HOT context,
liquidity proxies, low-price cost estimate, stop/cap ties, pending liquidation,
missing raw/cap coverage, held-day fit exclusion and learned-feature blacklist.
No post-result changes to features, model settings, thresholds, seeds or arms.
