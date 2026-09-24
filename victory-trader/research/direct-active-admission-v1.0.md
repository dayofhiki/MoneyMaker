# Request 153 — direct Active admission attribution

## Why this experiment follows request 152

Request 152 removed residual transport capacity as the main explanation for the
remaining Focus -> Active loss. The full-refresh cap=20 diagnostic had full
20/20 scoreable occupancy and zero sparse-slot waste, yet retained only 92.83%
of Focus crossings. The cap=8 policy retained 93.42%.

The next unresolved question is therefore which part of the handoff is wrong:
Focus support/eligibility, or the ranking that chooses the 20 Active names.

## Data discipline

This request opens no new date. It reuses only the already-opened request
151/152 block:

- 2026-05-21
- 2026-05-22
- 2026-05-26
- 2026-05-27
- 2026-05-28

All supervised models are fit only through 2026-01-16. No May outcome is used
for fitting.

## Attribution

For every runner crossing, classify the exact prior-minute state as:

1. unavailable to the causal pre-runner population;
2. available but outside Focus-60;
3. inside Focus-60 and rank 1-20;
4. inside Focus-60 and rank 21-40;
5. inside Focus-60 and rank 41-60.

Report these ranks under both the current admission score and the direct score.

## Candidate intervention

The current request-147 admission score is a product of two separately fitted
models:

`market_hazard_probability * next_minute_observability_probability`.

Request 153 tests one simpler target-aligned alternative: fit one causal model
directly to the unconditional event "this ticker has an exact next-minute
runner crossing." A missing exact-next-minute bar is a zero for this executable
event.

The direct model uses exactly the same BASELINE_FEATURES and the same
HistGradientBoosting hyperparameters as the existing market-wide models.

Focus-60 membership remains frozen under the existing market-hazard model.
Transport-survival fitting remains frozen. Active capacity remains 20.

For a fair operational comparison, the direct variant replaces:

- admission score with direct crossing probability;
- stale-incumbent retention value with
  `direct crossing probability * transport survival probability`.

Evaluate both legacy and direct policies at max additions 8 and 20.

## Preregistered interpretation

The key diagnostic is cap=20, because transport is allowed to fully converge to
the current desired top-20 set each minute.

- **direct_admission_candidate**: direct cap20 retention >=95% and improves on
  legacy cap20 by at least 1 percentage point.
- **partial_ranking_improvement**: direct improves cap20 by at least 1 point but
  remains below 95%.
- **direct_formulation_insufficient**: cap20 gain is less than 1 point.

Also compare average precision inside Focus and report cap=8 retention/by-day
results, but do not use those secondary metrics to rewrite the frozen diagnosis.

## Scope

This is development-only. It cannot promote a new admission policy and cannot
be called fresh validation. If the direct intervention is a candidate, the next
step is an integrated same-block check before freezing a later-session
validation request.
