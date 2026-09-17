from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .analytics import load_event_dataset


DEFAULT_DISCOVERY_FEATURES: tuple[str, ...] = (
    "previous_close",
    "event_volume",
    "cumulative_volume",
    "volume_5m",
    "volume_15m",
    "volume_30m",
    "volume_accel_1m_vs_prior20m",
    "volume_accel_5m_vs_prior20m",
    "rvol_cumulative_20d",
    "rvol_5m_20d",
    "vwap_distance_pct",
    "regular_vwap_distance_pct",
    "hod_distance_pct",
    "trailing_return_5m_pct",
    "trailing_return_15m_pct",
    "trailing_return_30m_pct",
    "volatility_5m_pct",
    "volatility_15m_pct",
    "volatility_30m_pct",
    "premarket_return_pct",
    "premarket_volume",
    "minutes_from_regular_open",
    "minutes_since_10pct_cross",
    "minutes_since_prior_threshold_cross",
    "prior_15m_high_distance_pct",
    "prior_15m_low_rebound_pct",
)


def _target_column(horizon_min: int, scenario: str) -> str:
    return f"return_{horizon_min}m_{scenario}_net_return_pct"


def _finite_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return float("nan"), float("nan")
    result = spearmanr(valid["x"], valid["y"])
    return float(result.statistic), float(result.pvalue)


def _bh_qvalues(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = pd.to_numeric(p_values, errors="coerce").dropna()
    if valid.empty:
        return result
    ordered = valid.sort_values()
    m = len(ordered)
    raw = ordered.to_numpy(dtype=float) * m / np.arange(1, m + 1, dtype=float)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def _prepare_feature_rows(
    frame: pd.DataFrame,
    feature: str,
    target_column: str,
) -> pd.DataFrame:
    required = {"trading_day", "threshold_pct", feature, target_column}
    missing = required - set(frame.columns)
    if missing:
        return pd.DataFrame()
    work = frame.loc[:, list(required)].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, target_column])
    if work.empty:
        return work
    work["trading_day"] = work["trading_day"].astype(str)
    work["threshold_pct"] = pd.to_numeric(work["threshold_pct"], errors="coerce")
    return work.dropna(subset=["threshold_pct"])


def _day_threshold_signal(
    work: pd.DataFrame,
    feature: str,
    target_column: str,
    *,
    min_cell_n: int = 5,
) -> tuple[float, float, int, int]:
    if work.empty:
        return float("nan"), float("nan"), 0, 0
    group_cols = ["trading_day", "threshold_pct"]
    eligible = work.groupby(group_cols)[feature].transform("count") >= min_cell_n
    cross = work.loc[eligible].copy()
    if cross.empty:
        return float("nan"), float("nan"), 0, 0
    cross["_feature_rank"] = cross.groupby(group_cols)[feature].rank(
        method="average", pct=True
    )
    cross["_target_residual"] = cross[target_column] - cross.groupby(group_cols)[
        target_column
    ].transform("mean")
    rho, p_value = _finite_spearman(cross["_feature_rank"], cross["_target_residual"])
    return rho, p_value, len(cross), cross["trading_day"].nunique()


def summarize_univariate_discovery(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
    features: tuple[str, ...] = DEFAULT_DISCOVERY_FEATURES,
    min_n: int = 100,
    min_days: int = 10,
) -> pd.DataFrame:
    """Screen point-in-time features while controlling for day and threshold.

    The signal statistic ranks a feature within each day/threshold cell and
    correlates it with return residuals from the same cell. This prevents a
    feature from looking useful merely because it is common on an unusually
    strong day or at an easier threshold.
    """
    target = _target_column(horizon_min, scenario)
    if target not in frame.columns:
        raise ValueError(f"dataset missing target column: {target}")

    rows: list[dict[str, float | int | str]] = []
    for feature in features:
        work = _prepare_feature_rows(frame, feature, target)
        if work.empty:
            continue
        rho, p_value, controlled_n, controlled_days = _day_threshold_signal(
            work, feature, target
        )
        direction = "higher" if pd.notna(rho) and rho >= 0 else "lower"

        work["_threshold_rank"] = work.groupby("threshold_pct")[feature].rank(
            method="average", pct=True
        )
        if direction == "higher":
            favorable = work.loc[work["_threshold_rank"] >= 0.8].copy()
            adverse = work.loc[work["_threshold_rank"] <= 0.2].copy()
        else:
            favorable = work.loc[work["_threshold_rank"] <= 0.2].copy()
            adverse = work.loc[work["_threshold_rank"] >= 0.8].copy()

        cell_cols = ["trading_day", "threshold_pct"]
        cell_means = work.groupby(cell_cols)[target].mean()

        def adjusted_mean(selection: pd.DataFrame) -> float:
            if selection.empty:
                return float("nan")
            index = pd.MultiIndex.from_frame(selection[cell_cols])
            benchmark = cell_means.reindex(index).to_numpy(dtype=float)
            return float((selection[target].to_numpy(dtype=float) - benchmark).mean())

        usable = len(work) >= min_n and work["trading_day"].nunique() >= min_days
        rows.append(
            {
                "feature": feature,
                "usable": int(usable),
                "n": len(work),
                "days": work["trading_day"].nunique(),
                "controlled_n": controlled_n,
                "controlled_days": controlled_days,
                "direction": direction,
                "day_threshold_spearman": rho,
                "p_value": p_value,
                "favorable_n": len(favorable),
                "favorable_mean_pct": float(favorable[target].mean())
                if not favorable.empty
                else float("nan"),
                "favorable_positive_rate": float((favorable[target] > 0).mean())
                if not favorable.empty
                else float("nan"),
                "favorable_p05_pct": float(favorable[target].quantile(0.05))
                if not favorable.empty
                else float("nan"),
                "favorable_adjusted_lift_pct": adjusted_mean(favorable),
                "adverse_n": len(adverse),
                "adverse_mean_pct": float(adverse[target].mean())
                if not adverse.empty
                else float("nan"),
                "raw_favorable_minus_adverse_pct": (
                    float(favorable[target].mean() - adverse[target].mean())
                    if not favorable.empty and not adverse.empty
                    else float("nan")
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value_bh"] = _bh_qvalues(result["p_value"])
    result["abs_spearman"] = result["day_threshold_spearman"].abs()
    return result.sort_values(
        ["usable", "q_value_bh", "abs_spearman"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)


def summarize_chronological_feature_holdout(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
    features: tuple[str, ...] = DEFAULT_DISCOVERY_FEATURES,
    train_fraction: float = 0.70,
    min_train_n: int = 100,
    min_test_n: int = 20,
) -> pd.DataFrame:
    """Learn one-sided feature cuts on early dates, score them on later dates.

    Cut direction and threshold-specific 20/80 percentiles are learned only
    from the chronological training slice. The later slice is never used to
    choose the cut, making this a small internal anti-overfitting check before
    moving to another month.
    """
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    target = _target_column(horizon_min, scenario)
    required = {"trading_day", "threshold_pct", target}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing holdout columns: {sorted(missing)}")

    base = frame.loc[pd.to_numeric(frame[target], errors="coerce").notna()].copy()
    base[target] = pd.to_numeric(base[target], errors="coerce")
    base["trading_day"] = base["trading_day"].astype(str)
    dates = sorted(base["trading_day"].unique())
    if len(dates) < 4:
        return pd.DataFrame()
    split = int(len(dates) * train_fraction)
    split = min(max(split, 2), len(dates) - 2)
    train_dates = set(dates[:split])
    test_dates = set(dates[split:])

    train_base = base.loc[base["trading_day"].isin(train_dates)]
    test_base = base.loc[base["trading_day"].isin(test_dates)].copy()
    benchmark = test_base.groupby(["trading_day", "threshold_pct"])[target].mean()

    rows: list[dict[str, float | int | str]] = []
    for feature in features:
        train = _prepare_feature_rows(train_base, feature, target)
        test = _prepare_feature_rows(test_base, feature, target)
        if train.empty or test.empty:
            continue
        rho, p_value, controlled_n, _ = _day_threshold_signal(train, feature, target)
        if pd.isna(rho):
            continue
        direction = "higher" if rho >= 0 else "lower"
        quantile = 0.8 if direction == "higher" else 0.2
        counts = train.groupby("threshold_pct")[feature].count()
        eligible_thresholds = counts.loc[counts >= 20].index
        cuts = (
            train.loc[train["threshold_pct"].isin(eligible_thresholds)]
            .groupby("threshold_pct")[feature]
            .quantile(quantile)
        )
        test["_cut"] = test["threshold_pct"].map(cuts)
        test = test.dropna(subset=["_cut"])
        if direction == "higher":
            selected = test.loc[test[feature] >= test["_cut"]].copy()
        else:
            selected = test.loc[test[feature] <= test["_cut"]].copy()

        adjusted_lift = float("nan")
        if not selected.empty:
            index = pd.MultiIndex.from_frame(selected[["trading_day", "threshold_pct"]])
            matched = benchmark.reindex(index).to_numpy(dtype=float)
            adjusted_lift = float(
                (selected[target].to_numpy(dtype=float) - matched).mean()
            )
        rows.append(
            {
                "feature": feature,
                "train_n": len(train),
                "test_feature_n": len(test),
                "train_direction": direction,
                "train_day_threshold_spearman": rho,
                "train_p_value": p_value,
                "train_controlled_n": controlled_n,
                "test_selected_n": len(selected),
                "test_selected_days": selected["trading_day"].nunique()
                if not selected.empty
                else 0,
                "test_selection_rate": len(selected) / len(test) if len(test) else float("nan"),
                "test_mean_pct": float(selected[target].mean())
                if not selected.empty
                else float("nan"),
                "test_positive_rate": float((selected[target] > 0).mean())
                if not selected.empty
                else float("nan"),
                "test_p05_pct": float(selected[target].quantile(0.05))
                if not selected.empty
                else float("nan"),
                "test_adjusted_lift_pct": adjusted_lift,
                "enough_sample": int(len(train) >= min_train_n and len(selected) >= min_test_n),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["train_q_value_bh"] = _bh_qvalues(result["train_p_value"])
    result["train_abs_spearman"] = result["train_day_threshold_spearman"].abs()
    return result.sort_values(
        ["enough_sample", "train_q_value_bh", "train_abs_spearman"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)


def render_edge_discovery(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> str:
    univariate = summarize_univariate_discovery(
        frame,
        horizon_min=horizon_min,
        scenario=scenario,
    )
    holdout = summarize_chronological_feature_holdout(
        frame,
        horizon_min=horizon_min,
        scenario=scenario,
    )
    sections = [
        "=== MoneyMaker Edge Discovery ===",
        f"target={_target_column(horizon_min, scenario)}",
        "",
        "=== Day/threshold-controlled univariate screen ===",
        (
            univariate.to_string(index=False)
            if not univariate.empty
            else "No eligible features available."
        ),
        "",
        "=== Chronological 70/30 feature holdout ===",
        (
            holdout.to_string(index=False)
            if not holdout.empty
            else "Not enough dated observations for holdout analysis."
        ),
    ]
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m victory_trader.edge_discovery")
    parser.add_argument("path", type=Path)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--scenario", default="base")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frame = load_event_dataset(args.path)
    report = render_edge_discovery(
        frame,
        horizon_min=args.horizon,
        scenario=args.scenario,
    )
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
