from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_unresolved_exposure(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> pd.DataFrame:
    """Quantify entered trades whose requested horizon cannot be observed/executed.

    A missing return after an executable entry is not a harmless missing value: the
    position existed, but the pre-specified exit could not be observed at the exact
    requested minute (for example because of a halt or the session boundary).

    ``break_even_unresolved_mean_pct`` answers the useful sensitivity question:
    what average net return across those unresolved exposures would make the total
    entered-sample mean exactly zero? It avoids silently conditioning expectancy on
    only the rows with observable exits.
    """
    if horizon_min <= 0:
        raise ValueError("horizon_min must be positive")

    value_column = f"return_{horizon_min}m_{scenario}_net_return_pct"
    required = {"threshold_pct", "entry_price", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing unresolved-exposure columns: {sorted(missing)}")

    rows: list[dict[str, float | int]] = []
    for threshold, group in frame.groupby("threshold_pct", sort=True):
        entry = pd.to_numeric(group["entry_price"], errors="coerce")
        values = pd.to_numeric(group[value_column], errors="coerce")
        entered = entry.notna()
        resolved = entered & values.notna()
        unresolved = entered & values.isna()

        entered_n = int(entered.sum())
        resolved_n = int(resolved.sum())
        unresolved_n = int(unresolved.sum())
        resolved_values = values.loc[resolved]
        resolved_sum = float(resolved_values.sum()) if resolved_n else 0.0

        break_even = (
            -resolved_sum / unresolved_n
            if unresolved_n
            else np.nan
        )
        worst_case_mean = (
            (resolved_sum - 100.0 * unresolved_n) / entered_n
            if entered_n
            else np.nan
        )
        zero_imputed_mean = (
            resolved_sum / entered_n
            if entered_n
            else np.nan
        )

        rows.append(
            {
                "threshold_pct": float(threshold),
                "horizon_min": int(horizon_min),
                "scenario": scenario,
                "event_rows": int(len(group)),
                "entered_n": entered_n,
                "resolved_n": resolved_n,
                "unresolved_after_entry_n": unresolved_n,
                "unresolved_after_entry_rate": (
                    unresolved_n / entered_n if entered_n else np.nan
                ),
                "observed_resolved_mean_pct": (
                    float(resolved_values.mean()) if resolved_n else np.nan
                ),
                "zero_imputed_all_entered_mean_pct": zero_imputed_mean,
                "worst_case_all_entered_mean_pct": worst_case_mean,
                "break_even_unresolved_mean_pct": break_even,
            }
        )

    return pd.DataFrame(rows)
