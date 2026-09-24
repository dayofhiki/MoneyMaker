# Request 155 — temporal Active admission

## Why this follows requests 152–154

Request 152 showed that raw transport capacity is not the residual bottleneck:
full refresh can keep Active 20/20 scoreable yet Focus -> Active retention stays
below 95%.

Request 153 showed that replacing the legacy product score with a direct
one-minute crossing model improves retention only slightly.

Request 154 decomposed the one-minute score family. A Focus-only direct model
improved cap=20 retention to 93.57% and Focus average precision to 0.1810, but
still failed the 95% target. More importantly, several variants continued to
show that the bounded cap=8 policy can beat full refresh, consistent with useful
temporal persistence rather than a need for faster attention switching.

## Hypothesis

Active-20 should not be trained only to answer "which ticker is most likely to
cross in the very next minute?" A trader managing attention should also retain a
ticker whose setup remains likely to mature over the next several minutes.

The primary hypothesis is therefore that a Focus-specific **3-minute crossing
probability** is a better Active admission value than the current one-minute
score.

## Data discipline

No new date is opened. Evaluation reuses only:

- 2026-05-21
- 2026-05-22
- 2026-05-26
- 2026-05-27
- 2026-05-28

All supervised fitting remains limited to dates through 2026-01-16.

## Frozen horizons

Train Focus-60-specific causal models for crossing within:

- 1 minute
- 2 minutes
- **3 minutes (primary)**
- 5 minutes

The 3-minute model is primary because prior transport-survival work and the
cap=8 persistence effect both point to a short multi-minute memory window. The
other horizons are sensitivities and cannot replace the primary diagnosis after
results are observed.

A negative label is used only when the entire requested wall-clock horizon fits
inside the regular session. A crossing inside the horizon is positive.

All models use the existing frozen baseline causal feature set and the same
HistGradientBoosting hyperparameters.

## Runtime comparison

Keep Focus-60, Active capacity 20, transport-survival model, and stale-incumbent
eviction accounting unchanged.

For each horizon, replace Active admission score with its crossing-within-H
probability and use:

`balanced_retention_value = temporal_probability * transport_survival_probability`.

Evaluate both cap=8 and cap=20. Record Focus -> Active retention and actual
subscription additions. The cap=8 result is primary because it represents the
bounded attention policy that already exhibited useful persistence.

## Preregistered primary diagnosis

Relative to legacy cap=8:

- **temporal_policy_candidate** if 3m cap=8 reaches >=95%, improves by >=1pp,
  and does not increase mean post-initial additions per decision.
- **temporal_ranking_candidate_allocator_mismatch** if 3m cap=20 reaches >=95%
  but 3m cap=8 does not.
- **partial_temporal_gain** if 3m cap=8 improves by >=1pp but remains below 95%.
- **temporal_target_insufficient** if the 3m cap=8 gain is <1pp.

## Scope

Development-only. No policy promotion and no later-session validation occurs
here. A passing temporal candidate must be integrated with downstream HOT
capture and then frozen before any later unopened dates are used.
