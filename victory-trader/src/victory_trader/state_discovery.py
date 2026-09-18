from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (5, 10, 15)
EPISODE_KEYS = ["trading_day", "ticker"]


def state_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    hod = pd.to_numeric(frame["hod_distance_pct"], errors="coerce")
    above = frame["above_regular_vwap"].astype("boolean").fillna(False)
    vol_ratio = pd.to_numeric(
        frame["volume_3m_vs_postcross_peak"], errors="coerce"
    )
    range_ratio = pd.to_numeric(
        frame["range_contraction_3m_vs_15m"], errors="coerce"
    )
    trailing3 = pd.to_numeric(frame["trailing_return_3m_pct"], errors="coerce")
    reclaim = (
        frame["reclaim_prior_5m_high_after_pullback"]
        .astype("boolean")
        .fillna(False)
    )
    new_hod = frame["new_hod"].astype("boolean").fillna(False)

    return {
        "reclaim_after_pullback": reclaim,
        "constructive_pullback": (
            hod.between(-4.0, -1.0, inclusive="both")
            & above
            & vol_ratio.le(0.60)
        ),
        "fresh_hod_impulse": new_hod & trailing3.gt(0.0) & above,
        "tight_consolidation": (
            hod.ge(-3.0)
            & above
            & range_ratio.le(0.65)
            & ~new_hod
        ),
        "failure_state": hod.le(-5.0) & ~above,
    }


def first_occurrences(
    frame: pd.DataFrame,
    mask: pd.Series,
) -> pd.DataFrame:
    selected = frame.loc[mask.fillna(False)].copy()
    if selected.empty:
        return selected
    return (
        selected.sort_values(EPISODE_KEYS + ["t"], kind="stable")
        .drop_duplicates(EPISODE_KEYS, keep="first")
        .reset_index(drop=True)
    )


def baseline_first_cross(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[
        pd.to_numeric(
            frame["minutes_since_10pct_cross"], errors="coerce"
        ).eq(0.0)
    ].copy()
    return (
        work.sort_values(EPISODE_KEYS + ["t"], kind="stable")
        .drop_duplicates(EPISODE_KEYS, keep="first")
        .reset_index(drop=True)
    )


def _summary_row(
    selected: pd.DataFrame,
    *,
    month: str,
    state: str,
    horizon: int,
    episode_total: int,
) -> dict[str, object]:
    gross_col = f"buy_return_{horizon}m_pct"
    base_col = f"buy_return_{horizon}m_base_net_return_pct"
    stress_col = f"buy_return_{horizon}m_stress_net_return_pct"
    required = {gross_col, base_col, stress_col}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"state panel missing outcome columns: {sorted(missing)}")

    gross = pd.to_numeric(selected[gross_col], errors="coerce")
    base = pd.to_numeric(selected[base_col], errors="coerce")
    stress = pd.to_numeric(selected[stress_col], errors="coerce")
    observed = gross.notna()
    obs = selected.loc[observed].copy()
    obs["_gross"] = gross.loc[observed]
    obs["_base"] = base.loc[observed]
    obs["_stress"] = stress.loc[observed]

    day_base = (
        obs.groupby("trading_day")["_base"].mean()
        if not obs.empty
        else pd.Series(dtype=float)
    )
    return {
        "month": month,
        "state": state,
        "horizon_min": horizon,
        "episode_total": int(episode_total),
        "signal_n": int(len(selected)),
        "episode_coverage": (
            float(len(selected) / episode_total) if episode_total else np.nan
        ),
        "observed_n": int(observed.sum()),
        "observed_rate": float(observed.mean()) if len(selected) else np.nan,
        "gross_mean_pct": (
            float(obs["_gross"].mean()) if not obs.empty else np.nan
        ),
        "gross_median_pct": (
            float(obs["_gross"].median()) if not obs.empty else np.nan
        ),
        "base_mean_pct": (
            float(obs["_base"].mean()) if not obs.empty else np.nan
        ),
        "base_median_pct": (
            float(obs["_base"].median()) if not obs.empty else np.nan
        ),
        "base_positive_rate": (
            float((obs["_base"] > 0).mean()) if not obs.empty else np.nan
        ),
        "base_p05_pct": (
            float(obs["_base"].quantile(0.05)) if not obs.empty else np.nan
        ),
        "stress_mean_pct": (
            float(obs["_stress"].mean()) if not obs.empty else np.nan
        ),
        "day_balanced_base_mean_pct": (
            float(day_base.mean()) if not day_base.empty else np.nan
        ),
        "days_with_signal": int(day_base.size),
        "mfe_15m_mean_pct": (
            float(
                pd.to_numeric(
                    obs.get("buy_mfe_15m_pct"), errors="coerce"
                ).mean()
            )
            if not obs.empty and "buy_mfe_15m_pct" in obs
            else np.nan
        ),
        "mae_15m_mean_pct": (
            float(
                pd.to_numeric(
                    obs.get("buy_mae_15m_pct"), errors="coerce"
                ).mean()
            )
            if not obs.empty and "buy_mae_15m_pct" in obs
            else np.nan
        ),
    }


def evaluate_month(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    episode_total = int(frame[EPISODE_KEYS].drop_duplicates().shape[0])
    selectors = state_masks(frame)
    groups = {"first_10_cross": baseline_first_cross(frame)}
    for name, mask in selectors.items():
        groups[name] = first_occurrences(frame, mask)

    rows: list[dict[str, object]] = []
    for state, selected in groups.items():
        for horizon in HORIZONS:
            rows.append(
                _summary_row(
                    selected,
                    month=month,
                    state=state,
                    horizon=horizon,
                    episode_total=episode_total,
                )
            )
    return pd.DataFrame(rows)


def summarize_stability(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state, horizon), group in details.groupby(
        ["state", "horizon_min"], sort=False
    ):
        valid = group.loc[group["observed_n"].ge(30)].copy()
        rows.append(
            {
                "state": state,
                "horizon_min": int(horizon),
                "months_tested": int(len(valid)),
                "min_observed_n": (
                    int(valid["observed_n"].min()) if len(valid) else 0
                ),
                "median_episode_coverage": (
                    float(valid["episode_coverage"].median())
                    if len(valid)
                    else np.nan
                ),
                "gross_positive_months": int(
                    (valid["gross_mean_pct"] > 0).sum()
                ),
                "base_positive_months": int(
                    (valid["base_mean_pct"] > 0).sum()
                ),
                "day_balanced_base_positive_months": int(
                    (valid["day_balanced_base_mean_pct"] > 0).sum()
                ),
                "median_gross_mean_pct": (
                    float(valid["gross_mean_pct"].median())
                    if len(valid)
                    else np.nan
                ),
                "median_base_mean_pct": (
                    float(valid["base_mean_pct"].median())
                    if len(valid)
                    else np.nan
                ),
                "worst_base_mean_pct": (
                    float(valid["base_mean_pct"].min())
                    if len(valid)
                    else np.nan
                ),
                "median_day_balanced_base_mean_pct": (
                    float(valid["day_balanced_base_mean_pct"].median())
                    if len(valid)
                    else np.nan
                ),
                "median_base_p05_pct": (
                    float(valid["base_p05_pct"].median())
                    if len(valid)
                    else np.nan
                ),
                "median_mfe_15m_pct": (
                    float(valid["mfe_15m_mean_pct"].median())
                    if len(valid)
                    else np.nan
                ),
                "median_mae_15m_pct": (
                    float(valid["mae_15m_mean_pct"].median())
                    if len(valid)
                    else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        [
            "base_positive_months",
            "day_balanced_base_positive_months",
            "median_base_mean_pct",
            "worst_base_mean_pct",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def render_report(
    details: pd.DataFrame,
    stability: pd.DataFrame,
) -> str:
    lines = [
        "=== MoneyMaker State Trader Discovery v0.1 ===",
        "unit=first occurrence of each pre-specified state per ticker-day episode",
        "development_months=already-seen January-March 2026 only",
        "entry=exact next-minute open; missing exact entry/target remains missing",
        "costs=existing light/base/stress model",
        "thresholds=pre-registered in research/state-trader-v0.1.md; not optimized on these results",
        "",
        "=== Cross-month stability ===",
        stability.to_string(index=False),
        "",
        "=== Month-by-month details ===",
        details.to_string(index=False),
    ]
    return "\n".join(lines)


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.state_discovery"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--stability-csv", type=Path, required=True)
    args = parser.parse_args()

    pieces = []
    for label, path in args.dataset:
        frame = pd.read_parquet(path)
        pieces.append(evaluate_month(frame, label))
    details = pd.concat(pieces, ignore_index=True)
    stability = summarize_stability(details)
    report = render_report(details, stability)
    print(report)

    for path in (args.report, args.details_csv, args.stability_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    stability.to_csv(args.stability_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
