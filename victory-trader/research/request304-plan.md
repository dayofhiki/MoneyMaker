# R304 preregistration — fixed reference and runtime common expiry

Execution status:62 selected tests and research lint passed. Full local run
exactly reproduced R303 decisions and corrected scores, all86 positions resolved,
and weighted training residuals were zero. The local economic gate FAILED:
B mean BASE -1.631033% versus A -1.397021%, with all four days negative.
Zero targets fell7029→1161/10619, confirming label expiry was removed without
establishing profitability. No design changes followed the local result; dispatch
the same implementation once through the existing consolidated workflow.

Design fixed before the first R304 replay. A is exact R303 B, trained on old
10min/2% trailing reference WAIT advantages with corrected weighted offsets.
B PRIMARY changes only the fixed reference's separate entry+10min expiry to the
runtime's EXISTING min(hot_t+60min, session close) cap. The 2% prefix drawdown
teacher, absorbing BASE-net -2.5% risk stop, raw regular next-print execution,
costs and86 frozen entries remain unchanged. No minimum holding period or forced
longer runtime hold. Runtime still evaluates score>0 at every completed observation.

Build B labels with existing continuation recursion, supplying a deterministic
causal +/-1 action for the SAME trailing rule. This selects common-cap recursion
without modifying any historical source function/default. It is a fixed teacher,
not a learned policy or peak oracle. Label futures never become runtime features.
Preserve original pi0 labels for common audits. Naturally zero outcomes are valid;
age>=10min alone must no longer manufacture zero WAIT value before common cap.

Refit identical 28 causal full-history features, outer three-day training folds,
seeds20263975+fold*100, tree settings/clipping/day-episode weights. Apply R303's
weighted original-target residual offset solely to training data. Refit corrected
historical baseline and reproduce saved R303 scores within1e-9 before comparison.
No fitted target or offset depends on excluded-day labels. No threshold search.

Dependencies R303 run36835089902 and R298 run36801235949. May5–8 development
only, zero market acquisition, June HOLD and later dates sealed. No entry filters,
no removal of cost-only stops or trajectories without eligible HOLD states.

Report label finite/zero/positive support before/after10min; both arms on common
old and common-cap labels; first/all/visited reference-self continuation audits.
Own self-policy labels are computed after fitting, never used for training or
selection. A and B still follow different policies from their fixed teacher;
aligned expiry does not establish optimality or policy consistency.

Economic report: all-entry BASE/LIGHT/STRESS, day means, median/CVaR5,
excluding-largest, gross/cost drag, risk-compatible upside, holds/exits,
first exits/feature support, execution delays and paired B-A/B-old intervals.
Gate: exact baseline replay/scores, zero weighted training residual,86 positions,
>=95% resolved, positive BASE mean,>=3 positive days, positive mean excluding
largest trade, positive paired gains against A/old and weighted learned-target
rank. Four reused days and upstream entry OOF not nested prohibit promotion.

Tests: removal of expiry-induced zero labels, identical trailing action before
expiry, cap/gap liquidation, absorbing risk before/tied with cap, both held-day
targets excluded from fit/calibration, baseline mismatch/future-day rejection.
Retain all inherited causal history, pending risk and calibration tests. Economic
failure must not be overridden by improved target support/rank or longer holds.
No post-result feature/threshold/seed/arm tuning; publish same design once.
