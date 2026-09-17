from __future__ import annotations

import pandas as pd


RVOL_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, float("inf")]
RVOL_LABELS = ["<1x", "1-2x", "2-5x", "5-10x", ">=10x"]


def _basic_row(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0,
            "mean_return_pct": float("nan"),
            "median_return_pct": float("nan"),
            "positive_rate": float("nan"),
            "p05_return_pct": float("nan"),
            "p95_return_pct": float("nan"),
        }
    return {
        "n": len(clean),
        "mean_return_pct": float(clean.mean()),
        "median_return_pct": float(clean.median()),
        "positive_rate": float((clean > 0).mean()),
        "p05_return_pct": float(clean.quantile(0.05)),
        "p95_return_pct": float(clean.quantile(0.95)),
    }


def summarize_rvol_net(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> pd.DataFrame:
    """Summarize RVOL using executable after-cost returns, never gross returns."""
    return_col = f"return_{horizon_min}m_{scenario}_net_return_pct"
    required = {"threshold_pct", "rvol_cumulative_20d", return_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing RVOL net analysis columns: {sorted(missing)}")

    work = frame.loc[:, list(required)].copy()
    work["rvol_cumulative_20d"] = pd.to_numeric(
        work["rvol_cumulative_20d"], errors="coerce"
    )
    work[return_col] = pd.to_numeric(work[return_col], errors="coerce")
    work = work.dropna(subset=["rvol_cumulative_20d", return_col])
    if work.empty:
        return pd.DataFrame()

    work["rvol_bucket"] = pd.cut(
        work["rvol_cumulative_20d"],
        bins=RVOL_BINS,
        labels=RVOL_LABELS,
        right=False,
        include_lowest=True,
    )
    rows: list[dict] = []
    for (threshold, bucket), group in work.groupby(
        ["threshold_pct", "rvol_bucket"], observed=True, sort=True
    ):
        rows.append(
            {
                "threshold_pct": float(threshold),
                "rvol_bucket": str(bucket),
                "horizon_min": horizon_min,
                "scenario": scenario,
                **_basic_row(group[return_col]),
            }
        )
    return pd.DataFrame(rows)
