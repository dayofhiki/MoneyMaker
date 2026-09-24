# Request 156 — dynamic absolute-threshold Active sizing

## Question

Does Active need to contain exactly 20 names every minute?

The current architecture forces a relative ranking decision: among Focus-60,
keep the best 20. That can create two symmetric errors:

1. quiet market: weak names are admitted merely because 20 slots exist;
2. crowded opportunity set: strong names below rank 20 are rejected merely
   because the slots are full.

Request 156 tests whether Active should instead be a variable-size set selected
by an **absolute three-minute signal threshold**.

## Scope and causal discipline

No new date is opened. Evaluation reuses only:

- 2026-05-21
- 2026-05-22
- 2026-05-26
- 2026-05-27
- 2026-05-28

All supervised fitting and all threshold derivation use dates through
2026-01-16 only.

Focus-60 remains frozen. This request isolates the Active-20 design question;
it does not yet test whether Focus itself should also become dynamic.

## Signal

Use the request-155 Focus-specific model for:

P(runner crossing within the next 3 wall-clock minutes).

No May outcome is used to choose a threshold.

## Threshold family

To avoid guessing raw probability cutoffs after seeing May, derive fixed
absolute score thresholds from the fit-period Focus score distribution at
quantiles:

- 25%
- 50%
- 65%
- 75%
- 85%
- 90%
- 95%

Each derived probability value is then frozen and applied unchanged at every
May timestamp. The selected Active set is every Focus name whose probability
is at or above that fixed value.

This is dynamic sizing, not per-minute ranking. The same numeric threshold can
produce 0 names in one minute and many names in another.

## Benchmark

The benchmark is pure Top-20 by the same three-minute probability. This keeps
the signal identical and changes only the selection rule.

## Metrics

For Top-20 and every threshold report:

- Focus -> Active exact-prior runner retention
- mean / median / p90 / p95 / min / max Active count
- fraction of decisions with zero Active names
- fraction of decisions with more than 20 Active names
- Active-count standard deviation
- three-minute positive rate among selected rows
- three-minute false-attention rate among selected rows

## Primary interpretation

Among threshold policies averaging no more than 20 Active names:

- **dynamic_threshold_reaches_target_with_lower_burden** if one reaches at
  least 95% retention while averaging fewer than 20 names.
- **dynamic_threshold_dominates_fixed20** if one matches or exceeds Top-20
  retention while averaging fewer than 20 names.
- **fixed20_not_refuted** if no tested fit-frozen threshold matches Top-20
  retention at mean Active <=20.

This is diagnostic only. Even a strong threshold result is not promoted until
temporal persistence and later unopened-date validation are tested.
