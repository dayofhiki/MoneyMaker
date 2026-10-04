# R311 — frozen Q/M transport to a causal sampled May population

Preregistered before outcome reads or second-data acquisition. R310 provides positive point estimates (same-day M-Q AUROC +.06128), with 7/8 checks passing but a confidence interval crossing zero. No further feature, C, seed, threshold, calibration or pipeline search on the nine first winners. This study freezes the model package and adds cases, without promoting a trading policy.

## Frozen training and references

Use official R310 run37218791436: original 97 May5–8 episodes,4,526 labeled states. Fit N/Q/M ONCE on all four development days using unchanged R309 fit_linear/predict_linear, fixed seed20264605. Inputs exactly match R310 packages (N5,Q9,M28). C=.1, L2, training-only weighted 1%/99% clipping, median/scaling/missing indicators; weight sum=97 episodes, no balancing/calibration. Training-derived first-snapshot prevalence is a constant prior reference. No May11–20 outcome fits or own-evaluation-day preprocessing. Preserve the original 86 entry benchmark and June HOLD/July–August seals.

## Additional population, selected without outcomes

Existing R233 run36251239308 artifact10909642550 contains the COMPLETE R163 validated-attention first-HOT population; R163 run36001945920 artifact10811471687 cross-checks candidate identities and input values. Read ONLY trading_day,ticker,t,bar_start_t,log_current_price,minutes_since_open,minutes_to_close,return_from_previous_close_pct from these files. Ignore every saved target, admission decision, learned economic score and outcome. Upstream attention fit boundary is Jan16 and second reranker fit Jan20–Feb9; completed-minute feature timestamp is bar_start_t+60,000, and upstream second features use t+1,000<=HOT. Reuse historical membership, not admission decisions fitted/calibrated on these dates.

Evaluation dates FIXED: May11,12,13,14,15,18,19,20 2026. Metadata-only audit before preregistration found 3,157 unique first-HOT ticker-days on these dates. Include identity if SHA256('R311|'+day+'|'+uppercase_ticker+'|'+HOT_ms).digest()[0]<64 (25% online-computable deterministic sampling). No outcome-based sampling, shortage replacement, global quota, future rank, candidate-price filter or stopping once enough winners are found. Persist the complete sampled cohort BEFORE fetching or labeling. Report full-day original/sample counts.

This is a BROADER-POPULATION stress/transport test: training97 were conditioned on an upstream OOF rich setup shortlist; eval sampling is independent of setup/economic labels. It is not an exact replication of that shortlist population, nor proof a failure here disproves the shortlist effect. May11–20 was explored in earlier project work and is additional development, NOT pristine untouched validation. Selection-population shift is explicitly reported.

## Causal observation and unchanged target

Fetch unadjusted one-second aggregates only for sampled May ticker-days via existing shared cache, with .02s request interval and standard existing retry logic. Log requests/cache/retries. Retain regular-session bars; all coverage gaps are preserved. One canonical observation per identity: completion time t+1,000 of the FIRST observed regular bar starting at/after HOT and before HOT+300,000. Require completion before min(HOT+60min,regular close). Do not require 3 future prints, skip an incomplete label to a later observation, choose a profitable delay, or replace missing cases.

Reuse original HOT-time completed-minute context fields. Q/M clock inputs use strictly completed seconds at observation; known cost drag uses last completed close. Missing history remains missing and is imputed ONLY with frozen May5–8 transform. Label uses unchanged R306 entry_labels: next regular open at/after decision, BASE-net -2.5% absorbing stop, common HOT+60min/session cap, pending terminal fill, costs. No fill beyond available session means unresolved, not a fabricated close fill. First observation must be selected before inspecting labels. Keep no-observation/incomplete-label rows and reasons in output. No model probabilities for unavailable observations; score available observations even if labels incomplete.

Save sampled cohort, scored states and compressed regular raw seconds as checkpoint artifacts. This potentially expensive data work must be reproducible if scratch disappears. Abort request scope outside fixed dates/identities; do not acquire June or new dates.

## Fixed evaluation/gate

Primary: first canonical observation, M-Q same-day positive-negative pair AUROC, original equal-day/episode weights and .5 ties. Also pooled AUROC/AP/Brier, constant training first-snapshot prior Brier, per-day positive counts/metrics, identity/observation/label coverage. Day and ticker-day bootstrap,1000 fixed draws with seeds20264650+cluster_type; preserve original weights; single-class/no-pair rank draws omitted/counted. Contrasts M-Q primary, Q-N coverage, M-N total. No all-state independence claim or threshold/trading-profit calculation.

Gate requires >=90% sampled identities have available observations and finite complete labels, at least20 positive episodes over at least4 positive days, M same-day/pool AUROC and AP better than Q, M Brier better than Q and training first prior, M day AUROC improving by>1e-9 on a majority of evaluable days, and ticker-day same-day M-Q gain interval lower bound>0. These are development research criteria, not executable promotion. Unresolved label population can bias metrics, so coverage/reasons stay visible even if gate passes. Do not alter sampling, date window, group definitions, C, endpoint or gate after outcomes.

Do not bypass old gate-dependent HOLD/promotion steps: this is additional exploratory observability coverage only. If transport fails, distinguish wider-population/first-observation distribution shift from loss of momentum increment; no rewrite of this cohort or winner filtering. Freeze this result before planning the next causal representation or population intervention. Realized net profitability remains untested.
