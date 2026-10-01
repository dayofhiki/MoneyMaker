# R303 preregistration — isolate weighted training residual offset

Design frozen before the first R303 replay. May5–8 development only. June HOLD
and later dates remain sealed. No market API acquisition or parameter search.

Execution status: implemented; 54 selected tests and repository research lint
passed. Full local replay exactly reproduced R302 B and saved pi1 scores in all
folds. Weighted training residual checks passed and all86 entries resolved.
Local B mean BASE -1.397021% versus A -1.449165%; all four days remain negative,
so the economic gate FAILED. No changes to the design followed those outcomes.
Dispatch the same frozen implementation once through the existing consolidated
research workflow; official artifacts remain authoritative.

R302 trees optimize day/episode-weighted squared loss, but their final additive
offset uses an unweighted mean residual. Test that specific inconsistency.

A: exact R302 B fixed-reference pi1 replay from saved predictions.
B PRIMARY: refit identical pi1 trees with the same 28 full-history features,
fixed pi0 WAIT targets, clipping, seeds 20263975+fold*100, and outer three days.
Replace only offset with the day/episode-weighted mean training residual.
As historically specified, residuals use ORIGINAL unclipped targets; clipping
still applies to tree fitting. Keep the sign threshold zero. No grid or held-day
calibration. The shared historical `_fit` implementation is unchanged.

Before correction, reproduce saved R302 B predictions within 1e-9. Reuse the
same tree object and feature columns. Verify held-day scores change only by one
constant per fold. Report old/new offsets, weighted residuals and sign changes.
Weighted residuals after correction must equal zero within 1e-9.

Keep all 86 entries / 97 candidates, frozen entry gates, costs, -2.5% BASE-net
stop, clock cap and absorbing pending next-print exit mechanics. Dependencies:
R302 run36833594641 and R298 run36801235949. Never discard execution gaps,
cost-only stops or positions without an eligible HOLD decision.

Report paired B-A and B-old means with day/ticker-day bootstrap intervals,
BASE/LIGHT/STRESS, daily means, median/CVaR5, excluding-largest result, gross
and cost drag, holds/exits, support and execution gaps. Recompute frozen-reference
and self-policy diagnostics AFTER fitting, never train or select using the latter.
A constant shift preserves within-fold ranks; pooled rank can change across
folds. Weighted training centering does not guarantee held-day calibration,
self-policy consistency or profits.

Gate: exact replay/prediction integrity, weighted residual zero, all entries,
>=95% resolution, positive BASE mean, >=3 positive days, positive mean without
largest trade, positive paired means against A/old, positive weighted learned
target ranking. Four reused days and upstream entry OOF not nested prohibit
production promotion even if the development gate passes. No result-driven
threshold/arm/feature/seed changes after local validation.

Tests cover unequal weights, tree identity/historical model preservation,
ineligible/missing-label exclusion, held-day target mutation, saved prediction
integrity and future-day rejection; retain inherited risk/causal/pending tests.
