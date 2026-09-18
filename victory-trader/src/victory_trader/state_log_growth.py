from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .state_action_risk import SEVERE_LOSS_PCT
from .state_action_value import (
    HORIZONS,
    _fit_affine_calibration,
    _label_eligible,
    _sample_training,
    _target_column,
    action_feature_frame,
    score_actions,
    train_direct_ev_model,
)
from .state_entry_ranker import EPISODE_KEYS, EV_GATE_PCT
from .state_value_model import (
    _chronological_fit_calibration_split,
    _eligible,
)


LOG_GROWTH_GATE = float(100.0 * np.log1p(EV_GATE_PCT / 100.0))
MIN_MONTH_TRADES = 15
BOOTSTRAP_SAMPLES = 10_000
POLICIES = ("earliest_ev_cap1", "log_growth_cap1")


@dataclass(frozen=True)
class LogGrowthModel:
    horizon: int
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    calibration_intercept: float
    calibration_slope: float


def arithmetic_to_log_growth(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    invalid = finite & (arr <= -100.0)
    if invalid.any():
        bad = arr[invalid][:5].tolist()
        raise ValueError(
            "finite base-net return <= -100% cannot be converted to log growth: "
            f"{bad}"
        )
    out = np.full(arr.shape, np.nan, dtype=float)
    out[finite] = 100.0 * np.log1p(arr[finite] / 100.0)
    return out


def train_log_growth_model(
    train: pd.DataFrame,
    horizon: int,
) -> LogGrowthModel:
    scoreable = train.loc[_eligible(train)].copy()
    fit_period, calibration = _chronological_fit_calibration_split(scoreable)

    fit = fit_period.loc[_label_eligible(fit_period, horizon)].copy()
    fit = _sample_training(fit)
    if fit.empty:
        raise ValueError(f"no training labels for {horizon}m log-growth model")

    x_full = action_feature_frame(fit)
    columns = [
        column for column in x_full.columns if x_full[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable log-growth features")

    arithmetic = pd.to_numeric(
        fit[_target_column(horizon)], errors="coerce"
    ).to_numpy(dtype=float)
    y_fit = arithmetic_to_log_growth(arithmetic)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20261220 + horizon,
    )
    model.fit(x_full.loc[:, columns], y_fit)

    cal = calibration.loc[_label_eligible(calibration, horizon)].copy()
    if cal.empty:
        intercept, slope = 0.0, 1.0
    else:
        x_cal = action_feature_frame(cal).reindex(columns=columns)
        raw_pred = model.predict(x_cal)
        actual_arithmetic = pd.to_numeric(
            cal[_target_column(horizon)], errors="coerce"
        ).to_numpy(dtype=float)
        actual_log_growth = arithmetic_to_log_growth(actual_arithmetic)
        intercept, slope = _fit_affine_calibration(
            raw_pred,
            actual_log_growth,
        )

    return LogGrowthModel(
        horizon=horizon,
        model=model,
        feature_columns=tuple(columns),
        calibration_intercept=intercept,
        calibration_slope=slope,
    )


def score_log_growth_actions(
    frame: pd.DataFrame,
    models: dict[int, LogGrowthModel],
) -> pd.DataFrame:
    scored = frame.loc[_eligible(frame)].copy()
    features = action_feature_frame(scored)
    columns: list[str] = []

    for horizon, fitted in models.items():
        raw = fitted.model.predict(
            features.reindex(columns=fitted.feature_columns)
        )
        calibrated = fitted.calibration_intercept + fitted.calibration_slope * raw
        column = f"predicted_log_growth_{horizon}m"
        scored[column] = calibrated
        columns.append(column)

    matrix = scored.loc[:, columns].to_numpy(dtype=float)
    best_index = np.nanargmax(matrix, axis=1)
    horizon_array = np.asarray(list(models.keys()), dtype=int)
    scored["predicted_best_log_growth"] = matrix[
        np.arange(len(scored)), best_index
    ]
    scored["predicted_best_log_growth_horizon_min"] = horizon_array[best_index]
    return scored


def score_both(
    frame: pd.DataFrame,
    direct_models: dict[int, object],
    log_models: dict[int, LogGrowthModel],
) -> pd.DataFrame:
    direct = score_actions(frame, direct_models)
    log = score_log_growth_actions(frame, log_models)
    for horizon in HORIZONS:
        direct[f"predicted_log_growth_{horizon}m"] = log[
            f"predicted_log_growth_{horizon}m"
        ].reindex(direct.index)
    direct["predicted_best_log_growth"] = log[
        "predicted_best_log_growth"
    ].reindex(direct.index)
    direct["predicted_best_log_growth_horizon_min"] = log[
        "predicted_best_log_growth_horizon_min"
    ].reindex(direct.index)
    return direct


def _signal_mask(scored: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "earliest_ev_cap1":
        return pd.to_numeric(
            scored["predicted_best_base_ev_pct"], errors="coerce"
        ).ge(EV_GATE_PCT)
    if policy == "log_growth_cap1":
        return pd.to_numeric(
            scored["predicted_best_log_growth"], errors="coerce"
        ).ge(LOG_GROWTH_GATE)
    raise ValueError(f"unknown policy: {policy}")


def _select_first_trades(
    scored: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    signals = scored.loc[_signal_mask(scored, policy)].copy()
    if signals.empty:
        return signals

    trades: list[dict[str, object]] = []
    for _, group in signals.groupby(EPISODE_KEYS, sort=False):
        row = group.sort_values("t", kind="stable").iloc[0]
        if policy == "earliest_ev_cap1":
            horizon = int(row["predicted_best_horizon_min"])
            decision_score = float(row["predicted_best_base_ev_pct"])
        elif policy == "log_growth_cap1":
            horizon = int(row["predicted_best_log_growth_horizon_min"])
            decision_score = float(row["predicted_best_log_growth"])
        else:
            raise ValueError(f"unknown policy: {policy}")

        entry = pd.to_numeric(
            pd.Series([row.get("entry_price", np.nan)]), errors="coerce"
        ).iloc[0]
        realized = pd.to_numeric(
            pd.Series([row.get(_target_column(horizon), np.nan)]),
            errors="coerce",
        ).iloc[0]

        # The first signal consumes the ticker-day attempt. Never use later
        # label availability or fillability to fall through to another state.
        if pd.isna(entry) or float(entry) <= 0 or pd.isna(realized):
            continue

        realized_value = float(realized)
        realized_log_growth = float(
            arithmetic_to_log_growth(np.asarray([realized_value]))[0]
        )
        item = row.to_dict()
        item["action_horizon_min"] = horizon
        item["selected_decision_score"] = decision_score
        item["realized_base_net_return_pct"] = realized_value
        item["realized_log_growth"] = realized_log_growth
        item["realized_gross_return_pct"] = float(
            row[f"buy_return_{horizon}m_pct"]
        )
        item["realized_stress_net_return_pct"] = float(
            row[f"buy_return_{horizon}m_stress_net_return_pct"]
        )
        trades.append(item)

    return pd.DataFrame(trades)


def _metrics(
    trades: pd.DataFrame,
    *,
    month: str,
    policy: str,
) -> dict[str, object]:
    if trades.empty:
        return {"month": month, "policy": policy, "trades": 0}

    base = pd.to_numeric(
        trades["realized_base_net_return_pct"], errors="coerce"
    )
    log_growth = pd.to_numeric(
        trades["realized_log_growth"], errors="coerce"
    )
    gross = pd.to_numeric(
        trades["realized_gross_return_pct"], errors="coerce"
    )
    stress = pd.to_numeric(
        trades["realized_stress_net_return_pct"], errors="coerce"
    )
    arithmetic_daily = (
        trades.assign(_base=base).groupby("trading_day")["_base"].mean()
    )
    log_daily = (
        trades.assign(_log=log_growth).groupby("trading_day")["_log"].mean()
    )
    horizons = pd.to_numeric(trades["action_horizon_min"], errors="coerce")

    return {
        "month": month,
        "policy": policy,
        "trades": int(len(trades)),
        "ticker_day_episodes": int(
            trades[EPISODE_KEYS].drop_duplicates().shape[0]
        ),
        "days": int(trades["trading_day"].nunique()),
        "trades_per_day": float(
            len(trades) / trades["trading_day"].nunique()
        ),
        "decision_score_mean": float(
            pd.to_numeric(
                trades["selected_decision_score"], errors="coerce"
            ).mean()
        ),
        "gross_mean_pct": float(gross.mean()),
        "base_mean_pct": float(base.mean()),
        "base_median_pct": float(base.median()),
        "base_positive_rate": float(base.gt(0).mean()),
        "actual_severe_loss_rate": float(base.le(SEVERE_LOSS_PCT).mean()),
        "base_p05_pct": float(base.quantile(0.05)),
        "stress_mean_pct": float(stress.mean()),
        "day_balanced_base_mean_pct": float(arithmetic_daily.mean()),
        "worst_day_mean_pct": float(arithmetic_daily.min()),
        "log_growth_mean": float(log_growth.mean()),
        "log_growth_median": float(log_growth.median()),
        "day_balanced_log_growth": float(log_daily.mean()),
        "worst_log_growth_day": float(log_daily.min()),
        "action_5m_rate": float(horizons.eq(5).mean()),
        "action_10m_rate": float(horizons.eq(10).mean()),
        "action_15m_rate": float(horizons.eq(15).mean()),
        "median_entry_minute": float(
            pd.to_numeric(
                trades["minutes_since_10pct_cross"], errors="coerce"
            ).median()
        ),
    }


def _common_episode_comparison(
    baseline: pd.DataFrame,
    log_trades: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    if baseline.empty or log_trades.empty:
        return {"month": month, "common_episodes": 0}

    columns = EPISODE_KEYS + [
        "t",
        "realized_base_net_return_pct",
        "realized_log_growth",
        "minutes_since_10pct_cross",
    ]
    merged = baseline.loc[:, columns].merge(
        log_trades.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_baseline", "_log"),
    )
    if merged.empty:
        return {"month": month, "common_episodes": 0}

    arithmetic_delta = (
        merged["realized_base_net_return_pct_log"]
        - merged["realized_base_net_return_pct_baseline"]
    )
    log_delta = (
        merged["realized_log_growth_log"]
        - merged["realized_log_growth_baseline"]
    )
    minute_delta = (
        merged["minutes_since_10pct_cross_log"]
        - merged["minutes_since_10pct_cross_baseline"]
    )
    return {
        "month": month,
        "common_episodes": int(len(merged)),
        "arithmetic_delta_mean_pct": float(arithmetic_delta.mean()),
        "log_growth_delta_mean": float(log_delta.mean()),
        "log_better_rate": float(log_delta.gt(0).mean()),
        "log_later_rate": float(minute_delta.gt(0).mean()),
        "median_entry_delay_minutes": float(minute_delta.median()),
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    comparisons: list[dict[str, object]] = []

    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )
        direct_models = {
            horizon: train_direct_ev_model(train, horizon)
            for horizon in HORIZONS
        }
        log_models = {
            horizon: train_log_growth_model(train, horizon)
            for horizon in HORIZONS
        }
        scored = score_both(holdout, direct_models, log_models)

        baseline = _select_first_trades(
            scored,
            policy="earliest_ev_cap1",
        )
        log_trades = _select_first_trades(
            scored,
            policy="log_growth_cap1",
        )
        selected = {
            "earliest_ev_cap1": baseline,
            "log_growth_cap1": log_trades,
        }

        for policy in POLICIES:
            trades = selected[policy]
            metric_rows.append(
                _metrics(
                    trades,
                    month=holdout_month,
                    policy=policy,
                )
            )
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["policy"] = policy
                trade_frames.append(item)

        comparisons.append(
            _common_episode_comparison(
                baseline,
                log_trades,
                month=holdout_month,
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
        pd.DataFrame(comparisons),
    )


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, group in details.groupby("policy", sort=True):
        valid = group.loc[group["trades"].ge(MIN_MONTH_TRADES)].copy()
        rows.append(
            {
                "policy": policy,
                "months_tested": int(len(valid)),
                "min_trades": int(valid["trades"].min())
                if len(valid)
                else 0,
                "base_positive_months": int(
                    valid["base_mean_pct"].gt(0).sum()
                ),
                "day_positive_months": int(
                    valid["day_balanced_base_mean_pct"].gt(0).sum()
                ),
                "log_positive_months": int(
                    valid["log_growth_mean"].gt(0).sum()
                ),
                "day_log_positive_months": int(
                    valid["day_balanced_log_growth"].gt(0).sum()
                ),
                "median_base_mean_pct": float(
                    valid["base_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "worst_base_mean_pct": float(
                    valid["base_mean_pct"].min()
                )
                if len(valid)
                else np.nan,
                "median_log_growth": float(
                    valid["log_growth_mean"].median()
                )
                if len(valid)
                else np.nan,
                "worst_log_growth": float(
                    valid["log_growth_mean"].min()
                )
                if len(valid)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def day_cluster_bootstrap(
    trades: pd.DataFrame,
    *,
    value_column: str,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float | int | str]:
    selected = trades.loc[trades["policy"].eq("log_growth_cap1")].copy()
    if selected.empty:
        return {
            "value_column": value_column,
            "days": 0,
            "day_balanced_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    daily = (
        selected.assign(
            _value=pd.to_numeric(
                selected[value_column], errors="coerce"
            )
        )
        .groupby("trading_day")["_value"]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    rng = np.random.default_rng(20261220)
    indices = rng.integers(0, len(daily), size=(samples, len(daily)))
    means = daily[indices].mean(axis=1)
    return {
        "value_column": value_column,
        "days": int(len(daily)),
        "day_balanced_mean": float(daily.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def success_check(
    details: pd.DataFrame,
    arithmetic_bootstrap: dict[str, object],
    log_bootstrap: dict[str, object],
) -> dict[str, bool]:
    baseline = details.loc[
        details["policy"].eq("earliest_ev_cap1")
    ].set_index("month")
    log_policy = details.loc[
        details["policy"].eq("log_growth_cap1")
    ].set_index("month")
    months = sorted(set(baseline.index) & set(log_policy.index))

    checks = {
        "enough_trades_every_month": bool(
            len(months) == 3
            and log_policy.loc[months, "trades"].ge(MIN_MONTH_TRADES).all()
        ),
        "base_positive_every_month": bool(
            len(months) == 3
            and log_policy.loc[months, "base_mean_pct"].gt(0).all()
        ),
        "day_base_positive_every_month": bool(
            len(months) == 3
            and log_policy.loc[
                months, "day_balanced_base_mean_pct"
            ].gt(0).all()
        ),
        "log_positive_every_month": bool(
            len(months) == 3
            and log_policy.loc[months, "log_growth_mean"].gt(0).all()
        ),
        "day_log_positive_every_month": bool(
            len(months) == 3
            and log_policy.loc[
                months, "day_balanced_log_growth"
            ].gt(0).all()
        ),
        "improves_day_log_every_month": bool(
            len(months) == 3
            and (
                log_policy.loc[months, "day_balanced_log_growth"]
                > baseline.loc[months, "day_balanced_log_growth"]
            ).all()
        ),
        "tail_and_stress_not_worse_every_month": bool(
            len(months) == 3
            and (
                log_policy.loc[months, "base_p05_pct"]
                >= baseline.loc[months, "base_p05_pct"]
            ).all()
            and (
                log_policy.loc[months, "stress_mean_pct"]
                >= baseline.loc[months, "stress_mean_pct"]
            ).all()
        ),
        "pooled_arithmetic_bootstrap_lower_bound_positive": bool(
            np.isfinite(float(arithmetic_bootstrap["ci_low"]))
            and float(arithmetic_bootstrap["ci_low"]) > 0
        ),
        "pooled_log_bootstrap_lower_bound_positive": bool(
            np.isfinite(float(log_bootstrap["ci_low"]))
            and float(log_bootstrap["ci_low"]) > 0
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    arithmetic_bootstrap: dict[str, object],
    log_bootstrap: dict[str, object],
    checks: dict[str, bool],
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State Log-Growth Action Value v0.8 ===",
            f"log_growth_gate={LOG_GROWTH_GATE:.9f}",
            "target=100*ln(1+realized_base_net_return_pct/100)",
            "target_winsorization=none",
            "evaluation=leave-one-month-out January-March 2026",
            "NOTE=development diagnostic only; no fresh month consumed",
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
            "=== Pooled arithmetic day-cluster bootstrap ===",
            pd.DataFrame([arithmetic_bootstrap]).to_string(index=False),
            "",
            "=== Pooled log-growth day-cluster bootstrap ===",
            pd.DataFrame([log_bootstrap]).to_string(index=False),
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
        prog="python -m victory_trader.state_log_growth"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    details, trades, comparisons = run_lomo(monthly)
    summary = summarize(details)
    arithmetic_bootstrap = day_cluster_bootstrap(
        trades,
        value_column="realized_base_net_return_pct",
    )
    log_bootstrap = day_cluster_bootstrap(
        trades,
        value_column="realized_log_growth",
    )
    checks = success_check(
        details,
        arithmetic_bootstrap,
        log_bootstrap,
    )
    report = render_report(
        details,
        summary,
        comparisons,
        arithmetic_bootstrap,
        log_bootstrap,
        checks,
    )
    print(report)

    paths = (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.summary_csv,
        args.comparisons_csv,
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
