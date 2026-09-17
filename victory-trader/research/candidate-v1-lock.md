# Candidate v1 Lock Record

Status: **FROZEN before February 2026 external validation**

## Provenance

- March enriched source run: `35284485307`
- March enriched artifact: `moneymaker-enriched-2026-03-v05`
- Candidate-score run: `35288983942`
- Candidate artifact: `moneymaker-candidate-2026-03-v1`
- Candidate artifact ID: `10525906017`
- Candidate artifact digest: `sha256:c9919597e0d6b00737365035af768df7c2cbb6342a15ae7d30a77543c6aa61d9`
- Locked rule JSON SHA-256: `3e8ffc4c0a26bfe59ba0bf738199bc90f639a1b745ee68daeeff4afd61d87c72`

## March split

- Fit dates: 2026-03-02 through 2026-03-20, 15 trading days
- Internal holdout: 2026-03-23 through 2026-03-31, 7 trading days
- Rule-eligible event thresholds: +10%, +20%, +30%, +50%
- +75% and +100% were not fit because the training sample was below the pre-specified minimum.

## Alpha score

Equal weights. Lower is better for all four components:

1. `prior_15m_low_rebound_pct`
2. `volatility_15m_pct`
3. `signal_bar_range_pct`
4. `trailing_return_5m_pct`

Each component is converted to a threshold-specific empirical percentile using the March fit slice. A row needs at least 3 of 4 alpha features. Candidate alpha selection is the top 20% of the equal-weight alpha score using March-fit-only cuts.

## Liquidity gate

All three must be at or above their threshold-specific March-fit 20th percentile:

1. `dollar_volume_5m`
2. `transactions_5m`
3. `active_minute_fraction_15m`

## Frozen-rule policy

- Do not change feature membership, directions, weights, percentile cutoffs, or minimum-feature count after inspecting February 2026.
- February is an external development check of this exact rule.
- Any later revision must be versioned as a new candidate and tested on a different untouched development sample.
- Validation May-June and final holdout July-August remain outside this tuning loop.

## March internal-holdout snapshot

The combined rule selected 246 executable events across all 7 holdout trading days, about 16.18% of executable events.

At 10 minutes:
- Gross mean: +0.3693%
- Base-net mean: -0.8670%
- Stress-net mean: -2.9066%
- Base-net matched lift versus same-day/same-threshold baseline: +0.7048 percentage points
- Exact-horizon observed-given-entry rate: 90.24%

This is evidence of ranking/selective value, not evidence of a profitable strategy. Candidate v1 therefore advances unchanged to February external validation.
