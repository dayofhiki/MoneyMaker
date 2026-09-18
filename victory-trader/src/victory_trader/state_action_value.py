from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .state_value_model import (
    MINUTE_MS,
    TRAIN_SAMPLE_EVERY_MINUTES,
    _chronological_fit_calibration_split,
    _eligible,
    feature_frame,
)


HORIZONS = (5, 10, 15)
EV_GATES_PCT = (0.0, 0.25, 0.50, 1.00)
EPISODE_KEYS = ["trading_day", "ticker"]
TARGET_WINSOR_LOW = 0.005
TARGET_WINSOR_HIGH = 0.995


@dataclass(frozen=True)
class DirectEVModel:
    horizon: int
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    target_low: float
    target_high: float
    calibration_intercept: float
    calibration_slope: float


def action_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = feature_frame(frame).copy()

    close = pd.to_numeric(frame.get("c"), errors="coerce")
    previous = pd.to_numeric(frame.get("previous_close"), errors="coerce")
    result["log_current_price"] = np.log(
        close.where(close > 0)
    )
    result["log_previous_close"] = np.log(
        previous.where(previous > 0)
    )
    return result


def _sample_training(frame: pd.DataFrame) -> pd.DataFrame:
    minutes = pd.to_numeric(
        frame["minutes_since_10pct_cross"], errors="coerce"
    )
    return frame.loc[
        minutes.mod(TRAIN_SAMPLE_EVERY_MINUTES).eq(0)
    ].copy()


def _target_column(horizon: int) -> str:
    return f"buy_return_{horizon}m_base_net_return_pct"


def _label_eligible(frame: pd.DataFrame, horizon: int) -> pd.Series:
    return (
        _eligible(frame)
        & pd.to_numeric(
            frame[_target_column(horizon)], errors="coerce"
        ).notna()
    )


def _fit_affine_calibration(
    prediction: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float]:
    valid = np.isfinite(prediction) & np.isfinite(actual)
    if valid.sum() < 100:
        return 0.0, 1.0

    x = prediction[valid].astype(float)
    y = actual[valid].astype(float)
    variance = float(np.var(x))
    if variance <= 1e-12:
        return float(np.mean(y) - np.mean(x)), 1.0

    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    slope = covariance / variance
    # Calibration is allowed to shrink/expand modestly, but not invert the
    # ranking learned by the state model.
    slope = float(np.clip(slope, 0.0, 1.5))
    intercept = float(np.mean(y) - slope * np.mean(x))
    intercept = float(np.clip(intercept, -5.0, 5.0))
    return intercept, slope


def train_direct_ev_model(
    train: pd.DataFrame,
    horizon: int,
) -> DirectEVModel:
    scoreable = train.loc[_eligible(train)].copy()
    fit_period, calibration = _chronological_fit_calibration_split(scoreable)

    fit = fit_period.loc[_label_eligible(fit_period, horizon)].copy()
    fit = _sample_training(fit)
    if fit.empty:
        raise ValueError(f"no training labels for {horizon}m direct EV model")

    x_full = action_feature_frame(fit)
    columns = [
        column for column in x_full.columns if x_full[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable direct-EV features")

    y = pd.to_numeric(
        fit[_target_column(horizon)], errors="coerce"
    )
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

    cal = calibration.loc[
        _label_eligible(calibration, horizon)
    ].copy()
    if cal.empty:
        intercept, slope = 0.0, 1.0
    else:
        x_cal = action_feature_frame(cal).reindex(columns=columns)
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


def score_actions(
    frame: pd.DataFrame,
    models: dict[int, DirectEVModel],
) -> pd.DataFrame:
    scored = frame.loc[_eligible(frame)].copy()
    features = action_feature_frame(scored)

    prediction_columns = []
    for horizon, fitted in models.items():
        raw = fitted.model.predict(
            features.reindex(columns=fitted.feature_columns)
        )
        calibrated = (
            fitted.calibration_intercept
            + fitted.calibration_slope * raw
        )
        column = f"predicted_base_ev_{horizon}m_pct"
        scored[column] = calibrated
        prediction_columns.append(column)

    matrix = scored.loc[:, prediction_columns].to_numpy(dtype=float)
    best_index = np.nanargmax(matrix, axis=1)
    horizon_array = np.asarray(list(models.keys()), dtype=int)
    scored["predicted_best_base_ev_pct"] = matrix[
        np.arange(len(scored)), best_index
    ]
    scored["predicted_best_horizon_min"] = horizon_array[best_index]
    return scored


def _non_overlapping_action_trades(
    scored: pd.DataFrame,
    *,
    ev_gate: float,
) -> pd.DataFrame:
    candidates = scored.loc[
        pd.to_numeric(
            scored["predicted_best_base_ev_pct"], errors="coerce"
        ).ge(ev_gate)
    ].copy()
    if candidates.empty:
        return candidates

    trades: list[dict[str, object]] = []
    for _, group in candidates.groupby(EPISODE_KEYS, sort=False):
        group = group.sort_values("t", kind="stable")
        next_allowed_t = -1
        for row in group.itertuples(index=False):
            t = int(row.t)
            if t < next_allowed_t:
                continue

            horizon = int(row.predicted_best_horizon_min)
            entry = pd.to_numeric(
                pd.Series([getattr(row, "entry_price", np.nan)]),
                errors="coerce",
            ).iloc[0]
            realized = pd.to_numeric(
                pd.Series(
                    [
                        getattr(
                            row,
                            _target_column(horizon),
                            np.nan,
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]
            if pd.isna(entry) or float(entry) <= 0 or pd.isna(realized):
                continue

            item = row._asdict()
            item["action_horizon_min"] = horizon
            item["realized_base_net_return_pct"] = float(realized)
            item["realized_gross_return_pct"] = float(
                getattr(row, f"buy_return_{horizon}m_pct")
            )
            item["realized_stress_net_return_pct"] = float(
                getattr(
                    row,
                    f"buy_return_{horizon}m_stress_net_return_pct",
                )
            )
            trades.append(item)
            next_allowed_t = t + horizon * MINUTE_MS

    return pd.DataFrame(trades)


def _metrics(
    trades: pd.DataFrame,
    *,
    month: str,
    ev_gate: float,
) -> dict[str, object]:
    if trades.empty:
        return {
            "month": month,
            "ev_gate_pct": ev_gate,
            "trades": 0,
        }

    base = pd.to_numeric(
        trades["realized_base_net_return_pct"], errors="coerce"
    )
    gross = pd.to_numeric(
        trades["realized_gross_return_pct"], errors="coerce"
    )
    stress = pd.to_numeric(
        trades["realized_stress_net_return_pct"], errors="coerce"
    )
    predicted = pd.to_numeric(
        trades["predicted_best_base_ev_pct"], errors="coerce"
    )
    daily = trades.assign(_base=base).groupby("trading_day")["_base"].mean()

    horizons = pd.to_numeric(
        trades["action_horizon_min"], errors="coerce"
    )
    return {
        "month": month,
        "ev_gate_pct": ev_gate,
        "trades": int(len(trades)),
        "ticker_day_episodes": int(
            trades[EPISODE_KEYS].drop_duplicates().shape[0]
        ),
        "days": int(trades["trading_day"].nunique()),
        "trades_per_day": float(
            len(trades) / trades["trading_day"].nunique()
        ),
        "predicted_base_ev_mean_pct": float(predicted.mean()),
        "gross_mean_pct": float(gross.mean()),
        "base_mean_pct": float(base.mean()),
        "base_median_pct": float(base.median()),
        "base_positive_rate": float((base > 0).mean()),
        "base_p05_pct": float(base.quantile(0.05)),
        "stress_mean_pct": float(stress.mean()),
        "day_balanced_base_mean_pct": float(daily.mean()),
        "worst_day_mean_pct": float(daily.min()),
        "action_5m_rate": float((horizons == 5).mean()),
        "action_10m_rate": float((horizons == 10).mean()),
        "action_15m_rate": float((horizons == 15).mean()),
        "median_entry_minute": float(
            pd.to_numeric(
                trades["minutes_since_10pct_cross"], errors="coerce"
            ).median()
        ),
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []

    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )
        models = {
            horizon: train_direct_ev_model(train, horizon)
            for horizon in HORIZONS
        }
        scored = score_actions(holdout, models)

        for gate in EV_GATES_PCT:
            trades = _non_overlapping_action_trades(
                scored,
                ev_gate=gate,
            )
            metrics_rows.append(
                _metrics(
                    trades,
                    month=holdout_month,
                    ev_gate=gate,
                )
            )
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["ev_gate_pct"] = gate
                trade_frames.append(item)

    return (
        pd.DataFrame(metrics_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
    )


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gate, group in details.groupby("ev_gate_pct", sort=True):
        valid = group.loc[group["trades"].ge(20)].copy()
        rows.append(
            {
                "ev_gate_pct": float(gate),
                "months_tested": int(len(valid)),
                "min_trades": int(valid["trades"].min()) if len(valid) else 0,
                "median_trades_per_day": float(
                    valid["trades_per_day"].median()
                )
                if len(valid)
                else np.nan,
                "gross_positive_months": int(
                    (valid["gross_mean_pct"] > 0).sum()
                ),
                "base_positive_months": int(
                    (valid["base_mean_pct"] > 0).sum()
                ),
                "day_positive_months": int(
                    (valid["day_balanced_base_mean_pct"] > 0).sum()
                ),
                "median_predicted_base_ev_pct": float(
                    valid["predicted_base_ev_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_gross_mean_pct": float(
                    valid["gross_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
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
                "median_entry_minute": float(
                    valid["median_entry_minute"].median()
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


def render_report(
    details: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker Direct Multi-Horizon Action Value v0.1 ===",
            "target=realized base-net return directly, not gross-return-minus-modeled-cost prediction",
            "actions=BUY_5M / BUY_10M / BUY_15M; choose the highest calibrated predicted base EV at every state",
            "features=point-in-time state + current absolute price + contemporaneous leave-one-out runner breadth",
            "training=train-only 0.5%/99.5% target winsorization; squared-error conditional-mean model",
            "calibration=train-month chronological 80/20 split with affine out-of-sample calibration",
            "evaluation=leave-one-month-out January-March 2026; non-overlapping repeated trades per ticker-day",
            "NOTE=development diagnostic only; no fresh month consumed",
            "",
            "=== Cross-month summary ===",
            summary.to_string(index=False),
            "",
            "=== Month-by-month ===",
            details.to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.state_action_value"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    details, trades = run_lomo(monthly)
    summary = summarize(details)
    report = render_report(details, summary)
    print(report)

    for path in (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.summary_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
