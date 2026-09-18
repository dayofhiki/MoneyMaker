from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from .state_action_value import (
    HORIZONS,
    TARGET_WINSOR_HIGH,
    TARGET_WINSOR_LOW,
    _fit_affine_calibration,
    _label_eligible,
    _sample_training,
    _target_column,
    action_feature_frame,
)
from .state_value_model import (
    MINUTE_MS,
    _chronological_fit_calibration_split,
    _eligible,
)


EV_GATE_PCT = 0.50
RISK_GATE_PROB = 0.20
SEVERE_LOSS_PCT = -5.0
EPISODE_KEYS = ["trading_day", "ticker"]
MIN_MONTH_TRADES = 15


@dataclass(frozen=True)
class PolicyVariant:
    name: str
    max_trades_per_episode: int | None
    risk_gate_prob: float | None


POLICIES = (
    PolicyVariant("repeat_no_risk", None, None),
    PolicyVariant("cap1_no_risk", 1, None),
    PolicyVariant("cap1_risk20", 1, RISK_GATE_PROB),
)


@dataclass(frozen=True)
class DirectActionRiskModel:
    horizon: int
    ev_model: HistGradientBoostingRegressor
    risk_model: HistGradientBoostingClassifier
    feature_columns: tuple[str, ...]
    target_low: float
    target_high: float
    calibration_intercept: float
    calibration_slope: float


def train_action_risk_model(
    train: pd.DataFrame,
    horizon: int,
) -> DirectActionRiskModel:
    scoreable = train.loc[_eligible(train)].copy()
    fit_period, calibration = _chronological_fit_calibration_split(scoreable)
    fit = fit_period.loc[_label_eligible(fit_period, horizon)].copy()
    fit = _sample_training(fit)
    if fit.empty:
        raise ValueError(f"no training labels for {horizon}m action-risk model")

    features = action_feature_frame(fit)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable action-risk features")
    x_fit = features.loc[:, columns]

    target = pd.to_numeric(fit[_target_column(horizon)], errors="coerce")
    low = float(target.quantile(TARGET_WINSOR_LOW))
    high = float(target.quantile(TARGET_WINSOR_HIGH))
    ev_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20260921 + horizon,
    )
    ev_model.fit(x_fit, target.clip(lower=low, upper=high))

    severe = target.le(SEVERE_LOSS_PCT).astype(int)
    if severe.nunique() < 2:
        raise ValueError(
            f"{horizon}m severe-loss labels contain fewer than two classes"
        )
    risk_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20260940 + horizon,
    )
    risk_model.fit(x_fit, severe)

    cal = calibration.loc[_label_eligible(calibration, horizon)].copy()
    if cal.empty:
        intercept, slope = 0.0, 1.0
    else:
        x_cal = action_feature_frame(cal).reindex(columns=columns)
        raw = ev_model.predict(x_cal)
        actual = pd.to_numeric(
            cal[_target_column(horizon)], errors="coerce"
        ).to_numpy(dtype=float)
        intercept, slope = _fit_affine_calibration(raw, actual)

    return DirectActionRiskModel(
        horizon=horizon,
        ev_model=ev_model,
        risk_model=risk_model,
        feature_columns=tuple(columns),
        target_low=low,
        target_high=high,
        calibration_intercept=intercept,
        calibration_slope=slope,
    )


def score_action_risk(
    frame: pd.DataFrame,
    models: dict[int, DirectActionRiskModel],
) -> pd.DataFrame:
    scored = frame.loc[_eligible(frame)].copy()
    features = action_feature_frame(scored)
    ev_columns: list[str] = []
    risk_columns: list[str] = []

    for horizon, fitted in models.items():
        x = features.reindex(columns=fitted.feature_columns)
        raw = fitted.ev_model.predict(x)
        ev_column = f"predicted_base_ev_{horizon}m_pct"
        risk_column = f"predicted_severe_loss_prob_{horizon}m"
        scored[ev_column] = (
            fitted.calibration_intercept
            + fitted.calibration_slope * raw
        )
        positive_index = list(fitted.risk_model.classes_).index(1)
        scored[risk_column] = fitted.risk_model.predict_proba(x)[
            :, positive_index
        ]
        ev_columns.append(ev_column)
        risk_columns.append(risk_column)

    ev_matrix = scored.loc[:, ev_columns].to_numpy(dtype=float)
    risk_matrix = scored.loc[:, risk_columns].to_numpy(dtype=float)
    best_index = np.nanargmax(ev_matrix, axis=1)
    rows = np.arange(len(scored))
    horizon_array = np.asarray(list(models.keys()), dtype=int)
    scored["predicted_best_base_ev_pct"] = ev_matrix[rows, best_index]
    scored["predicted_best_severe_loss_prob"] = risk_matrix[
        rows, best_index
    ]
    scored["predicted_best_horizon_min"] = horizon_array[best_index]
    return scored


def _select_trades(
    scored: pd.DataFrame,
    *,
    policy: PolicyVariant,
) -> pd.DataFrame:
    mask = pd.to_numeric(
        scored["predicted_best_base_ev_pct"], errors="coerce"
    ).ge(EV_GATE_PCT)
    if policy.risk_gate_prob is not None:
        mask &= pd.to_numeric(
            scored["predicted_best_severe_loss_prob"], errors="coerce"
        ).le(policy.risk_gate_prob)

    candidates = scored.loc[mask].copy()
    if candidates.empty:
        return candidates

    trades: list[dict[str, object]] = []
    for _, group in candidates.groupby(EPISODE_KEYS, sort=False):
        group = group.sort_values("t", kind="stable")
        next_allowed_t = -1
        episode_trades = 0
        for row in group.itertuples(index=False):
            if (
                policy.max_trades_per_episode is not None
                and episode_trades >= policy.max_trades_per_episode
            ):
                break
            t = int(row.t)
            if t < next_allowed_t:
                continue

            horizon = int(row.predicted_best_horizon_min)
            entry = pd.to_numeric(
                pd.Series([getattr(row, "entry_price", np.nan)]),
                errors="coerce",
            ).iloc[0]
            realized = pd.to_numeric(
                pd.Series([getattr(row, _target_column(horizon), np.nan)]),
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
            episode_trades += 1
            next_allowed_t = t + horizon * MINUTE_MS

    return pd.DataFrame(trades)


def _metrics(
    trades: pd.DataFrame,
    *,
    month: str,
    policy: PolicyVariant,
) -> dict[str, object]:
    if trades.empty:
        return {"month": month, "policy": policy.name, "trades": 0}

    base = pd.to_numeric(
        trades["realized_base_net_return_pct"], errors="coerce"
    )
    gross = pd.to_numeric(
        trades["realized_gross_return_pct"], errors="coerce"
    )
    stress = pd.to_numeric(
        trades["realized_stress_net_return_pct"], errors="coerce"
    )
    predicted_ev = pd.to_numeric(
        trades["predicted_best_base_ev_pct"], errors="coerce"
    )
    predicted_risk = pd.to_numeric(
        trades["predicted_best_severe_loss_prob"], errors="coerce"
    )
    daily = trades.assign(_base=base).groupby("trading_day")["_base"].mean()
    horizons = pd.to_numeric(trades["action_horizon_min"], errors="coerce")

    return {
        "month": month,
        "policy": policy.name,
        "trades": int(len(trades)),
        "ticker_day_episodes": int(
            trades[EPISODE_KEYS].drop_duplicates().shape[0]
        ),
        "days": int(trades["trading_day"].nunique()),
        "trades_per_day": float(
            len(trades) / trades["trading_day"].nunique()
        ),
        "predicted_base_ev_mean_pct": float(predicted_ev.mean()),
        "predicted_severe_loss_prob_mean": float(predicted_risk.mean()),
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


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
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
            horizon: train_action_risk_model(train, horizon)
            for horizon in HORIZONS
        }
        scored = score_action_risk(holdout, models)

        for policy in POLICIES:
            trades = _select_trades(scored, policy=policy)
            metric_rows.append(
                _metrics(trades, month=holdout_month, policy=policy)
            )
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["policy"] = policy.name
                trade_frames.append(item)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
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
                "worst_base_mean_pct": float(valid["base_mean_pct"].min())
                if len(valid)
                else np.nan,
                "median_day_balanced_base_mean_pct": float(
                    valid["day_balanced_base_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_actual_severe_loss_rate": float(
                    valid["actual_severe_loss_rate"].median()
                )
                if len(valid)
                else np.nan,
                "median_base_p05_pct": float(valid["base_p05_pct"].median())
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


def render_report(details: pd.DataFrame, summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "=== MoneyMaker Sequence Action Value + Severe-Loss Risk v0.2 ===",
            "features=point-in-time state + breadth + 36 causal ticker-local lags",
            "actions=BUY_5M / BUY_10M / BUY_15M selected by calibrated base EV",
            "entry_gate=predicted base EV >= 0.50% (pre-registered)",
            "risk=per-action probability of realized base-net return <= -5%",
            "risk_gate=predicted severe-loss probability <= 20% (pre-registered)",
            "policies=repeat/no-risk; one-entry-per-ticker-day/no-risk; one-entry-per-ticker-day/risk20",
            "evaluation=leave-one-month-out January-March 2026",
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
        prog="python -m victory_trader.state_action_risk"
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
        label: pd.read_parquet(path) for label, path in args.dataset
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
