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
    _sample_training,
    _target_column,
    action_feature_frame,
    score_actions,
    train_direct_ev_model,
)
from .state_value_model import (
    _chronological_fit_calibration_split,
    _eligible,
)


EV_GATE_PCT = 0.50
MINUTE_MS = 60_000
EPISODE_KEYS = ["trading_day", "ticker"]
MIN_MONTH_TRADES = 15
TARGET_WINSOR_LOW = 0.005
TARGET_WINSOR_HIGH = 0.995
POLICIES = ("earliest_ev_cap1", "one_step_stop_cap1")
BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class OneStepModel:
    horizon: int
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    target_low: float
    target_high: float
    calibration_intercept: float
    calibration_slope: float


def one_step_delta_target(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """Next-minute change in realized after-cost action value.

    Future information is used only here as a supervised label. A valid label
    requires the next row in the same ticker-day to be exactly one minute later.
    """

    target_col = _target_column(horizon)
    work = frame.loc[:, EPISODE_KEYS + ["t", target_col]].copy()
    work["_value"] = pd.to_numeric(work[target_col], errors="coerce")
    work["_t_num"] = pd.to_numeric(work["t"], errors="coerce")
    work["_original_index"] = work.index
    work = work.sort_values(EPISODE_KEYS + ["_t_num"], kind="stable")

    grouped = work.groupby(EPISODE_KEYS, sort=False)
    next_t = grouped["_t_num"].shift(-1)
    next_value = grouped["_value"].shift(-1)
    exact_next = next_t.sub(work["_t_num"]).eq(MINUTE_MS)
    delta = next_value.sub(work["_value"]).where(exact_next)

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[work["_original_index"].to_numpy()] = delta.to_numpy(dtype=float)
    return result


def train_one_step_model(
    train: pd.DataFrame,
    horizon: int,
) -> OneStepModel:
    scoreable = train.loc[_eligible(train)].copy()
    scoreable["_one_step_target"] = one_step_delta_target(
        scoreable, horizon
    )
    fit_period, calibration = _chronological_fit_calibration_split(scoreable)

    fit = fit_period.loc[
        pd.to_numeric(
            fit_period["_one_step_target"], errors="coerce"
        ).notna()
    ].copy()
    fit = _sample_training(fit)
    if fit.empty:
        raise ValueError(
            f"no training labels for {horizon}m one-step model"
        )

    features = action_feature_frame(fit)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable one-step features")

    y = pd.to_numeric(fit["_one_step_target"], errors="coerce")
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
        random_state=20261120 + horizon,
    )
    model.fit(features.loc[:, columns], y_fit)

    cal = calibration.loc[
        pd.to_numeric(
            calibration["_one_step_target"], errors="coerce"
        ).notna()
    ].copy()
    if cal.empty:
        intercept, slope = 0.0, 1.0
    else:
        raw_prediction = model.predict(
            action_feature_frame(cal).reindex(columns=columns)
        )
        actual = pd.to_numeric(
            cal["_one_step_target"], errors="coerce"
        ).to_numpy(dtype=float)
        intercept, slope = _fit_affine_calibration(raw_prediction, actual)

    return OneStepModel(
        horizon=horizon,
        model=model,
        feature_columns=tuple(columns),
        target_low=low,
        target_high=high,
        calibration_intercept=intercept,
        calibration_slope=slope,
    )


def score_one_step_models(
    frame: pd.DataFrame,
    models: dict[int, OneStepModel],
) -> pd.DataFrame:
    scoreable = frame.loc[_eligible(frame)].copy()
    features = action_feature_frame(scoreable)
    output = pd.DataFrame(index=scoreable.index)

    for horizon, fitted in models.items():
        raw = fitted.model.predict(
            features.reindex(columns=fitted.feature_columns)
        )
        calibrated = (
            fitted.calibration_intercept
            + fitted.calibration_slope * raw
        )
        output[f"predicted_one_step_delta_{horizon}m_pct"] = calibrated
    return output


def score_entry_states(
    frame: pd.DataFrame,
    direct_models: dict[int, object],
    one_step_models: dict[int, OneStepModel],
) -> pd.DataFrame:
    scored = score_actions(frame, direct_models)
    continuation = score_one_step_models(frame, one_step_models)
    columns = []
    for horizon in HORIZONS:
        column = f"predicted_one_step_delta_{horizon}m_pct"
        scored[column] = continuation[column].reindex(scored.index)
        columns.append(column)

    matrix = scored.loc[:, columns].to_numpy(dtype=float)
    horizons = pd.to_numeric(
        scored["predicted_best_horizon_min"], errors="coerce"
    ).to_numpy(dtype=int)
    horizon_to_index = {h: i for i, h in enumerate(HORIZONS)}
    indices = np.asarray(
        [horizon_to_index[int(h)] for h in horizons], dtype=int
    )
    scored["predicted_selected_one_step_delta_pct"] = matrix[
        np.arange(len(scored)), indices
    ]
    return scored


def _policy_signal_mask(
    scored: pd.DataFrame,
    *,
    policy: str,
) -> pd.Series:
    ev_ok = pd.to_numeric(
        scored["predicted_best_base_ev_pct"], errors="coerce"
    ).ge(EV_GATE_PCT)
    if policy == "earliest_ev_cap1":
        return ev_ok
    if policy == "one_step_stop_cap1":
        delta = pd.to_numeric(
            scored["predicted_selected_one_step_delta_pct"],
            errors="coerce",
        )
        return ev_ok & delta.le(0.0)
    raise ValueError(f"unknown one-step policy: {policy}")


def _select_first_trades(
    scored: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    signals = scored.loc[_policy_signal_mask(scored, policy=policy)].copy()
    if signals.empty:
        return signals

    trades: list[dict[str, object]] = []
    for _, group in signals.groupby(EPISODE_KEYS, sort=False):
        row = group.sort_values("t", kind="stable").iloc[0]
        horizon = int(row["predicted_best_horizon_min"])
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
        item["selected_predicted_base_ev_pct"] = float(
            row["predicted_best_base_ev_pct"]
        )
        item["selected_predicted_one_step_delta_pct"] = float(
            row["predicted_selected_one_step_delta_pct"]
        )
        item["realized_base_net_return_pct"] = float(realized)
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
    gross = pd.to_numeric(
        trades["realized_gross_return_pct"], errors="coerce"
    )
    stress = pd.to_numeric(
        trades["realized_stress_net_return_pct"], errors="coerce"
    )
    daily = trades.assign(_base=base).groupby("trading_day")["_base"].mean()
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
        "predicted_base_ev_mean_pct": float(
            pd.to_numeric(
                trades["selected_predicted_base_ev_pct"], errors="coerce"
            ).mean()
        ),
        "predicted_one_step_delta_mean_pct": float(
            pd.to_numeric(
                trades["selected_predicted_one_step_delta_pct"],
                errors="coerce",
            ).mean()
        ),
        "gross_mean_pct": float(gross.mean()),
        "base_mean_pct": float(base.mean()),
        "base_median_pct": float(base.median()),
        "base_positive_rate": float(base.gt(0).mean()),
        "actual_severe_loss_rate": float(base.le(SEVERE_LOSS_PCT).mean()),
        "base_p05_pct": float(base.quantile(0.05)),
        "stress_mean_pct": float(stress.mean()),
        "day_balanced_base_mean_pct": float(daily.mean()),
        "worst_day_mean_pct": float(daily.min()),
        "action_5m_rate": float(horizons.eq(5).mean()),
        "action_10m_rate": float(horizons.eq(10).mean()),
        "action_15m_rate": float(horizons.eq(15).mean()),
        "median_entry_minute": float(
            pd.to_numeric(
                trades["minutes_since_10pct_cross"], errors="coerce"
            ).median()
        ),
    }


def _diagnostic_metrics(
    prediction: pd.Series,
    actual: pd.Series,
    mask: pd.Series,
    *,
    month: str,
    pool: str,
    horizon: str,
) -> dict[str, object]:
    valid = (
        mask
        & pd.to_numeric(prediction, errors="coerce").notna()
        & pd.to_numeric(actual, errors="coerce").notna()
    )
    p = pd.to_numeric(prediction.loc[valid], errors="coerce")
    y = pd.to_numeric(actual.loc[valid], errors="coerce")
    return {
        "month": month,
        "pool": pool,
        "horizon": horizon,
        "rows": int(len(y)),
        "predicted_mean_pct": float(p.mean()) if len(p) else np.nan,
        "realized_mean_pct": float(y.mean()) if len(y) else np.nan,
        "spearman": float(p.corr(y, method="spearman"))
        if len(y) >= 2
        else np.nan,
        "sign_agreement": float(p.le(0).eq(y.le(0)).mean())
        if len(y)
        else np.nan,
        "predicted_wait_rate": float(p.gt(0).mean())
        if len(p)
        else np.nan,
        "realized_wait_rate": float(y.gt(0).mean())
        if len(y)
        else np.nan,
    }


def _continuation_diagnostics(
    scored: pd.DataFrame,
    *,
    month: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ev_ok = pd.to_numeric(
        scored["predicted_best_base_ev_pct"], errors="coerce"
    ).ge(EV_GATE_PCT)
    chosen = pd.to_numeric(
        scored["predicted_best_horizon_min"], errors="coerce"
    )

    pooled_p: list[pd.Series] = []
    pooled_y: list[pd.Series] = []

    for horizon in HORIZONS:
        actual = one_step_delta_target(scored, horizon)
        prediction = pd.to_numeric(
            scored[f"predicted_one_step_delta_{horizon}m_pct"],
            errors="coerce",
        )
        all_mask = pd.Series(True, index=scored.index)
        selected_mask = ev_ok & chosen.eq(horizon)
        rows.append(
            _diagnostic_metrics(
                prediction,
                actual,
                all_mask,
                month=month,
                pool="all_states",
                horizon=str(horizon),
            )
        )
        rows.append(
            _diagnostic_metrics(
                prediction,
                actual,
                selected_mask,
                month=month,
                pool="ev_selected",
                horizon=str(horizon),
            )
        )

        valid = selected_mask & prediction.notna() & actual.notna()
        pooled_p.append(prediction.loc[valid])
        pooled_y.append(actual.loc[valid])

    if pooled_p:
        p = pd.concat(pooled_p, ignore_index=True)
        y = pd.concat(pooled_y, ignore_index=True)
    else:
        p = pd.Series(dtype=float)
        y = pd.Series(dtype=float)

    rows.append(
        {
            "month": month,
            "pool": "ev_selected_all_horizons",
            "horizon": "selected",
            "rows": int(len(y)),
            "predicted_mean_pct": float(p.mean()) if len(p) else np.nan,
            "realized_mean_pct": float(y.mean()) if len(y) else np.nan,
            "spearman": float(p.corr(y, method="spearman"))
            if len(y) >= 2
            else np.nan,
            "sign_agreement": float(p.le(0).eq(y.le(0)).mean())
            if len(y)
            else np.nan,
            "predicted_wait_rate": float(p.gt(0).mean())
            if len(p)
            else np.nan,
            "realized_wait_rate": float(y.gt(0).mean())
            if len(y)
            else np.nan,
        }
    )
    return pd.DataFrame(rows)


def _common_episode_comparison(
    baseline: pd.DataFrame,
    stopped: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    if baseline.empty or stopped.empty:
        return {"month": month, "common_episodes": 0}
    columns = EPISODE_KEYS + [
        "t",
        "realized_base_net_return_pct",
        "minutes_since_10pct_cross",
    ]
    merged = baseline.loc[:, columns].merge(
        stopped.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_baseline", "_stop"),
    )
    if merged.empty:
        return {"month": month, "common_episodes": 0}

    delta = (
        merged["realized_base_net_return_pct_stop"]
        - merged["realized_base_net_return_pct_baseline"]
    )
    minute_delta = (
        merged["minutes_since_10pct_cross_stop"]
        - merged["minutes_since_10pct_cross_baseline"]
    )
    return {
        "month": month,
        "common_episodes": int(len(merged)),
        "stop_minus_baseline_base_mean_pct": float(delta.mean()),
        "stop_better_rate": float(delta.gt(0).mean()),
        "stop_later_rate": float(minute_delta.gt(0).mean()),
        "median_entry_delay_minutes": float(minute_delta.median()),
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
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
        one_step_models = {
            horizon: train_one_step_model(train, horizon)
            for horizon in HORIZONS
        }
        scored = score_entry_states(
            holdout, direct_models, one_step_models
        )
        diagnostics.append(
            _continuation_diagnostics(scored, month=holdout_month)
        )

        selected: dict[str, pd.DataFrame] = {}
        for policy in POLICIES:
            trades = _select_first_trades(scored, policy=policy)
            selected[policy] = trades
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
                selected["earliest_ev_cap1"],
                selected["one_step_stop_cap1"],
                month=holdout_month,
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
        pd.concat(diagnostics, ignore_index=True)
        if diagnostics
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
                "median_day_balanced_base_mean_pct": float(
                    valid["day_balanced_base_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_base_p05_pct": float(
                    valid["base_p05_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_stress_mean_pct": float(
                    valid["stress_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "base_positive_months",
            "day_positive_months",
            "median_base_mean_pct",
            "worst_base_mean_pct",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def day_cluster_bootstrap(
    trades: pd.DataFrame,
    *,
    policy: str = "one_step_stop_cap1",
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float | int]:
    selected = trades.loc[trades["policy"].eq(policy)].copy()
    if selected.empty:
        return {
            "days": 0,
            "day_balanced_mean_pct": np.nan,
            "ci_low_pct": np.nan,
            "ci_high_pct": np.nan,
        }

    daily = (
        selected.assign(
            _base=pd.to_numeric(
                selected["realized_base_net_return_pct"], errors="coerce"
            )
        )
        .groupby("trading_day")["_base"]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    rng = np.random.default_rng(20261120)
    indices = rng.integers(0, len(daily), size=(samples, len(daily)))
    means = daily[indices].mean(axis=1)
    return {
        "days": int(len(daily)),
        "day_balanced_mean_pct": float(daily.mean()),
        "ci_low_pct": float(np.quantile(means, 0.025)),
        "ci_high_pct": float(np.quantile(means, 0.975)),
    }


def success_check(
    details: pd.DataFrame,
    diagnostics: pd.DataFrame,
    bootstrap: dict[str, float | int],
) -> dict[str, bool]:
    baseline = details.loc[
        details["policy"].eq("earliest_ev_cap1")
    ].set_index("month")
    stopped = details.loc[
        details["policy"].eq("one_step_stop_cap1")
    ].set_index("month")
    months = sorted(set(baseline.index) & set(stopped.index))

    enough = (
        len(months) == 3
        and stopped.loc[months, "trades"].ge(MIN_MONTH_TRADES).all()
    )
    base_positive = (
        len(months) == 3
        and stopped.loc[months, "base_mean_pct"].gt(0).all()
    )
    day_positive = (
        len(months) == 3
        and stopped.loc[
            months, "day_balanced_base_mean_pct"
        ].gt(0).all()
    )
    improves_day = (
        len(months) == 3
        and (
            stopped.loc[months, "day_balanced_base_mean_pct"]
            > baseline.loc[months, "day_balanced_base_mean_pct"]
        ).all()
    )
    tail_not_worse = (
        len(months) == 3
        and (
            stopped.loc[months, "base_p05_pct"]
            >= baseline.loc[months, "base_p05_pct"]
        ).all()
        and (
            stopped.loc[months, "stress_mean_pct"]
            >= baseline.loc[months, "stress_mean_pct"]
        ).all()
    )

    diag = diagnostics.loc[
        diagnostics["pool"].eq("ev_selected_all_horizons")
    ].set_index("month")
    positive_spearman = (
        len(months) == 3
        and set(months).issubset(diag.index)
        and diag.loc[months, "spearman"].gt(0).all()
    )
    bootstrap_positive = bool(
        np.isfinite(float(bootstrap["ci_low_pct"]))
        and float(bootstrap["ci_low_pct"]) > 0
    )

    checks = {
        "enough_trades_every_month": bool(enough),
        "base_positive_every_month": bool(base_positive),
        "day_positive_every_month": bool(day_positive),
        "improves_day_every_month": bool(improves_day),
        "tail_and_stress_not_worse_every_month": bool(tail_not_worse),
        "ev_selected_delta_spearman_positive_every_month": bool(
            positive_spearman
        ),
        "pooled_day_bootstrap_lower_bound_positive": bool(
            bootstrap_positive
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
    bootstrap: dict[str, float | int],
    checks: dict[str, bool],
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State One-Step Stop v0.5 ===",
            "buy_gate=calibrated direct best base EV >= 0.50%",
            "continuation_target_h=realized base-net h-minute action at t+1m minus t; exact next minute only",
            "stop_policy=first EV-qualified state whose chosen horizon has predicted one-step delta <= 0.00%; one attempt per ticker-day",
            "features=point-in-time state + breadth + 36 causal ticker-local lags",
            "evaluation=leave-one-month-out January-March 2026",
            "NOTE=development diagnostic only; no fresh month consumed",
            "",
            "=== Cross-month summary ===",
            summary.to_string(index=False),
            "",
            "=== Month-by-month ===",
            details.to_string(index=False),
            "",
            "=== Holdout continuation diagnostics ===",
            diagnostics.to_string(index=False),
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
        prog="python -m victory_trader.state_one_step_stop"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    details, trades, diagnostics, comparisons = run_lomo(monthly)
    summary = summarize(details)
    bootstrap = day_cluster_bootstrap(trades)
    checks = success_check(details, diagnostics, bootstrap)
    report = render_report(
        details,
        summary,
        diagnostics,
        comparisons,
        bootstrap,
        checks,
    )
    print(report)

    for path in (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.summary_csv,
        args.diagnostics_csv,
        args.comparisons_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    diagnostics.to_csv(args.diagnostics_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
