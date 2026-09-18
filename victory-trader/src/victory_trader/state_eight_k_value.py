from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .state_action_value import (
    HORIZONS,
    TARGET_WINSOR_HIGH,
    TARGET_WINSOR_LOW,
    DirectEVModel,
    _fit_affine_calibration,
    _label_eligible,
    _sample_training,
    _target_column,
    action_feature_frame,
    score_actions,
    train_direct_ev_model,
)
from .state_eight_k_enrichment import EIGHT_K_FEATURES
from .state_entry_ranker import EPISODE_KEYS, EV_GATE_PCT
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


POLICIES = ("earliest_ev_cap1", "eight_k_ev_cap1")


def eight_k_action_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = action_feature_frame(frame).copy()
    for column in EIGHT_K_FEATURES:
        if column in frame.columns:
            result[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            result[column] = pd.Series(np.nan, index=frame.index, dtype=float)
    return result


def train_eight_k_ev_model(
    train: pd.DataFrame,
    horizon: int,
) -> DirectEVModel:
    scoreable = train.loc[_eligible(train)].copy()
    fit_period, calibration = _chronological_fit_calibration_split(scoreable)

    fit = fit_period.loc[_label_eligible(fit_period, horizon)].copy()
    fit = _sample_training(fit)
    if fit.empty:
        raise ValueError(f"no training labels for {horizon}m 8-K EV model")

    x_full = eight_k_action_feature_frame(fit)
    columns = [
        column for column in x_full.columns if x_full[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable 8-K EV features")

    y = pd.to_numeric(fit[_target_column(horizon)], errors="coerce")
    low = float(y.quantile(TARGET_WINSOR_LOW))
    high = float(y.quantile(TARGET_WINSOR_HIGH))
    y_fit = y.clip(lower=low, upper=high)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
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
        intercept, slope = 0.0, 1.0
    else:
        x_cal = eight_k_action_feature_frame(cal).reindex(columns=columns)
        raw_pred = model.predict(x_cal)
        actual = pd.to_numeric(
            cal[_target_column(horizon)], errors="coerce"
        ).to_numpy(dtype=float)
        intercept, slope = _fit_affine_calibration(raw_pred, actual)

    return DirectEVModel(
        horizon=horizon,
        model=model,
        feature_columns=tuple(columns),
        target_low=low,
        target_high=high,
        calibration_intercept=intercept,
        calibration_slope=slope,
    )


def score_eight_k_actions(
    frame: pd.DataFrame,
    models: dict[int, DirectEVModel],
) -> pd.DataFrame:
    scored = frame.loc[_eligible(frame)].copy()
    features = eight_k_action_feature_frame(scored)
    columns: list[str] = []
    horizons = list(models.keys())

    for horizon, fitted in models.items():
        raw = fitted.model.predict(
            features.reindex(columns=fitted.feature_columns)
        )
        calibrated = (
            fitted.calibration_intercept
            + fitted.calibration_slope * raw
        )
        column = f"predicted_eight_k_ev_{horizon}m_pct"
        scored[column] = calibrated
        columns.append(column)

    matrix = scored.loc[:, columns].to_numpy(dtype=float)
    best_index = np.nanargmax(matrix, axis=1)
    horizon_array = np.asarray(horizons, dtype=int)
    scored["predicted_best_eight_k_ev_pct"] = matrix[
        np.arange(len(scored)), best_index
    ]
    scored["predicted_best_eight_k_horizon_min"] = horizon_array[best_index]
    return scored


def score_both(
    frame: pd.DataFrame,
    baseline_models: dict[int, DirectEVModel],
    eight_k_models: dict[int, DirectEVModel],
) -> pd.DataFrame:
    baseline = score_actions(frame, baseline_models)
    augmented = score_eight_k_actions(frame, eight_k_models)

    for horizon in HORIZONS:
        column = f"predicted_eight_k_ev_{horizon}m_pct"
        baseline[column] = augmented[column].reindex(baseline.index)
    baseline["predicted_best_eight_k_ev_pct"] = augmented[
        "predicted_best_eight_k_ev_pct"
    ].reindex(baseline.index)
    baseline["predicted_best_eight_k_horizon_min"] = augmented[
        "predicted_best_eight_k_horizon_min"
    ].reindex(baseline.index)
    return baseline


def _policy_columns(policy: str) -> tuple[str, str]:
    if policy == "earliest_ev_cap1":
        return "predicted_best_base_ev_pct", "predicted_best_horizon_min"
    if policy == "eight_k_ev_cap1":
        return (
            "predicted_best_eight_k_ev_pct",
            "predicted_best_eight_k_horizon_min",
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
    augmented: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    if baseline.empty or augmented.empty:
        return {"month": month, "common_episodes": 0}

    columns = EPISODE_KEYS + [
        "realized_base_net_return_pct",
        "minutes_since_10pct_cross",
    ]
    merged = baseline.loc[:, columns].merge(
        augmented.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_baseline", "_eight_k"),
    )
    if merged.empty:
        return {"month": month, "common_episodes": 0}

    delta = (
        merged["realized_base_net_return_pct_eight_k"]
        - merged["realized_base_net_return_pct_baseline"]
    )
    minute_delta = (
        merged["minutes_since_10pct_cross_eight_k"]
        - merged["minutes_since_10pct_cross_baseline"]
    )
    return {
        "month": month,
        "common_episodes": int(len(merged)),
        "eight_k_minus_baseline_base_mean_pct": float(delta.mean()),
        "eight_k_better_rate": float(delta.gt(0).mean()),
        "eight_k_later_rate": float(minute_delta.gt(0).mean()),
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
        scoreable = holdout.loc[_eligible(holdout)].copy()
        ticker_days = scoreable[
            [
                "ticker",
                "trading_day",
                "eight_k_query_complete",
                "eight_k_unique_filings_30d",
                "eight_k_has_filing_7d",
            ]
        ].drop_duplicates(["ticker", "trading_day"])
        coverage_rows.append(
            {
                "month": holdout_month,
                "scoreable_rows": int(len(scoreable)),
                "scoreable_ticker_days": int(len(ticker_days)),
                "query_complete_coverage": float(
                    pd.to_numeric(
                        ticker_days["eight_k_query_complete"],
                        errors="coerce",
                    ).eq(1.0).mean()
                ),
                "filing_30d_prevalence": float(
                    pd.to_numeric(
                        ticker_days["eight_k_unique_filings_30d"],
                        errors="coerce",
                    ).gt(0).mean()
                ),
                "filing_7d_prevalence": float(
                    pd.to_numeric(
                        ticker_days["eight_k_has_filing_7d"],
                        errors="coerce",
                    ).eq(1.0).mean()
                ),
            }
        )

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
        eight_k_models = {
            horizon: train_eight_k_ev_model(train, horizon)
            for horizon in HORIZONS
        }
        scored = score_both(holdout, baseline_models, eight_k_models)

        baseline = select_first_trades(scored, policy="earliest_ev_cap1")
        augmented = select_first_trades(scored, policy="eight_k_ev_cap1")
        selected = {
            "earliest_ev_cap1": baseline,
            "eight_k_ev_cap1": augmented,
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
                augmented,
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
    augmented = details.loc[
        details["policy"].eq("eight_k_ev_cap1")
    ].set_index("month")
    months = sorted(set(baseline.index) & set(augmented.index))

    checks = {
        "enough_trades_every_month": bool(
            len(months) == 3
            and augmented.loc[months, "trades"].ge(MIN_MONTH_TRADES).all()
        ),
        "base_positive_every_month": bool(
            len(months) == 3
            and augmented.loc[months, "base_mean_pct"].gt(0).all()
        ),
        "day_positive_every_month": bool(
            len(months) == 3
            and augmented.loc[
                months, "day_balanced_base_mean_pct"
            ].gt(0).all()
        ),
        "improves_day_vs_baseline_every_month": bool(
            len(months) == 3
            and (
                augmented.loc[months, "day_balanced_base_mean_pct"]
                > baseline.loc[months, "day_balanced_base_mean_pct"]
            ).all()
        ),
        "tail_and_stress_not_worse_than_baseline_every_month": bool(
            len(months) == 3
            and (
                augmented.loc[months, "base_p05_pct"]
                >= baseline.loc[months, "base_p05_pct"]
            ).all()
            and (
                augmented.loc[months, "stress_mean_pct"]
                >= baseline.loc[months, "stress_mean_pct"]
            ).all()
        ),
        "query_complete_coverage_100pct_every_month": bool(
            len(coverage) == 3
            and coverage["query_complete_coverage"].eq(1.0).all()
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
            "=== MoneyMaker State 8-K Disclosure Action Value v1.1 ===",
            "entry_gate=predicted base-net EV >= 0.50%",
            "8-K window=strictly prior 30 calendar days",
            "evaluation=leave-one-month-out January-March 2026",
            "NOTE=development diagnostic only; no fresh month consumed",
            "",
            "=== Enrichment coverage ===",
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
        prog="python -m victory_trader.state_eight_k_value"
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
        policy="eight_k_ev_cap1",
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
