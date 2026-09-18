from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .state_action_value import (
    HORIZONS,
    TARGET_WINSOR_HIGH,
    TARGET_WINSOR_LOW,
    _label_eligible,
    _sample_training,
    _target_column,
    score_actions,
    train_direct_ev_model,
)
from .state_entry_ranker import EPISODE_KEYS, EV_GATE_PCT
from .state_multi_source_value import (
    _coverage_row,
    multi_source_action_feature_frame,
)
from .state_rank_turn import day_cluster_bootstrap
from .state_short_volume_value import (
    BOOTSTRAP_SAMPLES,
    MIN_MONTH_TRADES,
    _metrics,
    summarize,
)
from .state_value_model import (
    _chronological_fit_calibration_split,
    _eligible,
)


@dataclass(frozen=True)
class MedianActionModel:
    horizon: int
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    target_low: float
    target_high: float
    calibration_offset: float


POLICIES = ("earliest_ev_cap1", "median_multi_source_cap1")


def _median_calibration_offset(
    prediction: np.ndarray,
    actual: np.ndarray,
) -> float:
    valid = np.isfinite(prediction) & np.isfinite(actual)
    if valid.sum() < 100:
        return 0.0
    return float(np.median(actual[valid] - prediction[valid]))


def train_median_multi_source_model(
    train: pd.DataFrame,
    horizon: int,
) -> MedianActionModel:
    scoreable = train.loc[_eligible(train)].copy()
    fit_period, calibration = _chronological_fit_calibration_split(scoreable)

    fit = fit_period.loc[_label_eligible(fit_period, horizon)].copy()
    fit = _sample_training(fit)
    if fit.empty:
        raise ValueError(
            f"no training labels for {horizon}m median multi-source model"
        )

    x_full = multi_source_action_feature_frame(fit)
    columns = [
        column for column in x_full.columns if x_full[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable median multi-source features")

    y = pd.to_numeric(fit[_target_column(horizon)], errors="coerce")
    low = float(y.quantile(TARGET_WINSOR_LOW))
    high = float(y.quantile(TARGET_WINSOR_HIGH))
    y_fit = y.clip(lower=low, upper=high)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20260921 + horizon,
    )
    model.fit(x_full.loc[:, columns], y_fit)

    cal = calibration.loc[_label_eligible(calibration, horizon)].copy()
    if cal.empty:
        offset = 0.0
    else:
        x_cal = multi_source_action_feature_frame(cal).reindex(
            columns=columns
        )
        raw_pred = model.predict(x_cal)
        actual = pd.to_numeric(
            cal[_target_column(horizon)], errors="coerce"
        ).to_numpy(dtype=float)
        offset = _median_calibration_offset(raw_pred, actual)

    return MedianActionModel(
        horizon=horizon,
        model=model,
        feature_columns=tuple(columns),
        target_low=low,
        target_high=high,
        calibration_offset=offset,
    )


def score_median_actions(
    frame: pd.DataFrame,
    models: dict[int, MedianActionModel],
) -> pd.DataFrame:
    scored = frame.loc[_eligible(frame)].copy()
    features = multi_source_action_feature_frame(scored)
    prediction_columns: list[str] = []
    horizons = list(models.keys())

    for horizon, fitted in models.items():
        raw = fitted.model.predict(
            features.reindex(columns=fitted.feature_columns)
        )
        calibrated = raw + fitted.calibration_offset
        column = f"predicted_median_{horizon}m_pct"
        scored[column] = calibrated
        prediction_columns.append(column)

    matrix = scored.loc[:, prediction_columns].to_numpy(dtype=float)
    best_index = np.nanargmax(matrix, axis=1)
    horizon_array = np.asarray(horizons, dtype=int)
    scored["predicted_best_median_pct"] = matrix[
        np.arange(len(scored)), best_index
    ]
    scored["predicted_best_median_horizon_min"] = horizon_array[best_index]
    return scored


def score_both(
    frame: pd.DataFrame,
    baseline_models,
    median_models: dict[int, MedianActionModel],
) -> pd.DataFrame:
    baseline = score_actions(frame, baseline_models)
    median = score_median_actions(frame, median_models)
    for horizon in HORIZONS:
        column = f"predicted_median_{horizon}m_pct"
        baseline[column] = median[column].reindex(baseline.index)
    baseline["predicted_best_median_pct"] = median[
        "predicted_best_median_pct"
    ].reindex(baseline.index)
    baseline["predicted_best_median_horizon_min"] = median[
        "predicted_best_median_horizon_min"
    ].reindex(baseline.index)
    return baseline


def _policy_columns(policy: str) -> tuple[str, str]:
    if policy == "earliest_ev_cap1":
        return "predicted_best_base_ev_pct", "predicted_best_horizon_min"
    if policy == "median_multi_source_cap1":
        return (
            "predicted_best_median_pct",
            "predicted_best_median_horizon_min",
        )
    raise ValueError(f"unknown policy: {policy}")


def select_first_trades(
    scored: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    score_column, horizon_column = _policy_columns(policy)
    signals = scored.loc[
        pd.to_numeric(scored[score_column], errors="coerce").ge(EV_GATE_PCT)
    ].copy()
    if signals.empty:
        return signals

    trades: list[dict[str, object]] = []
    for _, group in signals.groupby(EPISODE_KEYS, sort=False):
        row = group.sort_values("t", kind="stable").iloc[0]
        horizon = int(row[horizon_column])
        decision_score = float(row[score_column])

        entry = pd.to_numeric(
            pd.Series([row.get("entry_price", np.nan)]), errors="coerce"
        ).iloc[0]
        realized = pd.to_numeric(
            pd.Series([row.get(_target_column(horizon), np.nan)]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(entry) or float(entry) <= 0 or pd.isna(realized):
            continue

        item = row.to_dict()
        item["action_horizon_min"] = horizon
        item["selected_decision_score"] = decision_score
        item["realized_base_net_return_pct"] = float(realized)
        item["realized_gross_return_pct"] = float(
            row[f"buy_return_{horizon}m_pct"]
        )
        item["realized_stress_net_return_pct"] = float(
            row[f"buy_return_{horizon}m_stress_net_return_pct"]
        )
        trades.append(item)

    return pd.DataFrame(trades)


def compare_policies(
    baseline: pd.DataFrame,
    median: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    if baseline.empty or median.empty:
        return {"month": month, "common_episodes": 0}

    columns = EPISODE_KEYS + [
        "realized_base_net_return_pct",
        "minutes_since_10pct_cross",
    ]
    merged = baseline.loc[:, columns].merge(
        median.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_baseline", "_median"),
    )
    if merged.empty:
        return {"month": month, "common_episodes": 0}

    delta = (
        merged["realized_base_net_return_pct_median"]
        - merged["realized_base_net_return_pct_baseline"]
    )
    minute_delta = (
        merged["minutes_since_10pct_cross_median"]
        - merged["minutes_since_10pct_cross_baseline"]
    )
    return {
        "month": month,
        "common_episodes": int(len(merged)),
        "median_minus_baseline_base_mean_pct": float(delta.mean()),
        "median_better_rate": float(delta.gt(0).mean()),
        "median_later_rate": float(minute_delta.gt(0).mean()),
        "median_entry_delay_minutes": float(minute_delta.median()),
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    comparisons: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for holdout_month, holdout in monthly_frames.items():
        coverage_rows.append(_coverage_row(holdout, holdout_month))
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )
        baseline_models = {
            horizon: train_direct_ev_model(train, horizon)
            for horizon in HORIZONS
        }
        median_models = {
            horizon: train_median_multi_source_model(train, horizon)
            for horizon in HORIZONS
        }
        scored = score_both(holdout, baseline_models, median_models)

        baseline = select_first_trades(scored, policy="earliest_ev_cap1")
        median = select_first_trades(
            scored,
            policy="median_multi_source_cap1",
        )
        selected = {
            "earliest_ev_cap1": baseline,
            "median_multi_source_cap1": median,
        }

        for policy, trades in selected.items():
            metric_rows.append(
                _metrics(trades, month=holdout_month, policy=policy)
            )
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["policy"] = policy
                trade_frames.append(item)

        comparisons.append(
            compare_policies(
                baseline,
                median,
                month=holdout_month,
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
        pd.DataFrame(comparisons),
        pd.DataFrame(coverage_rows),
    )


def success_check(
    details: pd.DataFrame,
    coverage: pd.DataFrame,
    bootstrap: dict[str, float | int],
) -> dict[str, bool]:
    baseline = details.loc[
        details["policy"].eq("earliest_ev_cap1")
    ].set_index("month")
    median = details.loc[
        details["policy"].eq("median_multi_source_cap1")
    ].set_index("month")
    months = sorted(set(baseline.index) & set(median.index))

    checks = {
        "enough_trades_every_month": bool(
            len(months) == 3
            and median.loc[months, "trades"].ge(MIN_MONTH_TRADES).all()
        ),
        "base_positive_every_month": bool(
            len(months) == 3
            and median.loc[months, "base_mean_pct"].gt(0).all()
        ),
        "day_positive_every_month": bool(
            len(months) == 3
            and median.loc[
                months, "day_balanced_base_mean_pct"
            ].gt(0).all()
        ),
        "improves_day_vs_baseline_every_month": bool(
            len(months) == 3
            and (
                median.loc[months, "day_balanced_base_mean_pct"]
                > baseline.loc[months, "day_balanced_base_mean_pct"]
            ).all()
        ),
        "tail_and_stress_not_worse_than_baseline_every_month": bool(
            len(months) == 3
            and (
                median.loc[months, "base_p05_pct"]
                >= baseline.loc[months, "base_p05_pct"]
            ).all()
            and (
                median.loc[months, "stress_mean_pct"]
                >= baseline.loc[months, "stress_mean_pct"]
            ).all()
        ),
        "short_volume_coverage_at_least_90pct_every_month": bool(
            len(coverage) == 3
            and coverage["short_volume_latest_coverage"].ge(0.90).all()
        ),
        "short_interest_and_8k_coverage_pass_every_month": bool(
            len(coverage) == 3
            and coverage["short_interest_latest_coverage"].ge(0.90).all()
            and coverage["eight_k_query_complete_coverage"].eq(1.0).all()
        ),
        "pooled_day_bootstrap_lower_bound_positive": bool(
            np.isfinite(float(bootstrap["ci_low_pct"]))
            and float(bootstrap["ci_low_pct"]) > 0
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    coverage: pd.DataFrame,
    bootstrap: dict[str, float | int],
    checks: dict[str, bool],
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State Robust Median Action Value v1.4 ===",
            "entry_gate=predicted median base-net return >= 0.50%",
            "loss=absolute_error",
            "calibration=median residual offset only",
            "external_features=frozen v1.3 multi-source set",
            "evaluation=leave-one-month-out January-March 2026",
            "NOTE=adaptive development diagnostic only; no fresh month consumed",
            "",
            "=== External feature coverage ===",
            coverage.to_string(index=False),
            "",
            "=== Cross-month summary ===",
            summary.to_string(index=False),
            "",
            "=== Month-by-month ===",
            details.to_string(index=False),
            "",
            "=== Common-episode comparison ===",
            comparisons.to_string(index=False),
            "",
            "=== Pooled day-cluster bootstrap ===",
            pd.DataFrame([bootstrap]).to_string(index=False),
            "",
            "=== Pre-registered success check ===",
            pd.DataFrame(
                [
                    {"criterion": key, "pass": value}
                    for key, value in checks.items()
                ]
            ).to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.state_robust_median_value"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    details, trades, comparisons, coverage = run_lomo(monthly)
    summary = summarize(details)
    bootstrap = day_cluster_bootstrap(
        trades,
        policy="median_multi_source_cap1",
        samples=BOOTSTRAP_SAMPLES,
    )
    checks = success_check(details, coverage, bootstrap)
    report = render_report(
        details,
        summary,
        comparisons,
        coverage,
        bootstrap,
        checks,
    )
    print(report)

    for path in (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.summary_csv,
        args.comparisons_csv,
        args.coverage_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    coverage.to_csv(args.coverage_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
