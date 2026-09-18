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

from .state_value_model import (
    MINUTE_MS,
    TRAIN_SAMPLE_EVERY_MINUTES,
    _base_scenario,
    _chronological_fit_calibration_split,
    _eligible,
    _label_eligible,
    _modeled_net_from_predicted_gross,
    feature_frame,
)


ENTRY_HORIZON = 10
RISK_HORIZON = 15
EV_GATES_PCT = (0.0, 0.25, 0.50)
RISK_GATE_PROB = 0.20
SEVERE_MAE_PCT = -5.0


@dataclass(frozen=True)
class EVRiskFold:
    alpha_model: HistGradientBoostingRegressor
    risk_model: HistGradientBoostingClassifier
    alpha_columns: tuple[str, ...]
    risk_columns: tuple[str, ...]
    train_target_low: float
    train_target_high: float


def _sample_training(frame: pd.DataFrame) -> pd.DataFrame:
    minutes = pd.to_numeric(
        frame["minutes_since_10pct_cross"], errors="coerce"
    )
    return frame.loc[
        minutes.mod(TRAIN_SAMPLE_EVERY_MINUTES).eq(0)
    ].copy()


def _usable_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns if frame[column].notna().any()
    ]


def train_fold(train: pd.DataFrame) -> EVRiskFold:
    scoreable = train.loc[_eligible(train)].copy()
    fit_period, _ = _chronological_fit_calibration_split(scoreable)

    alpha_fit = fit_period.loc[
        _label_eligible(fit_period, ENTRY_HORIZON)
    ].copy()
    alpha_fit = _sample_training(alpha_fit)

    x_alpha_full = feature_frame(alpha_fit)
    alpha_columns = _usable_columns(x_alpha_full)
    if not alpha_columns:
        raise ValueError("no usable alpha features")
    x_alpha = x_alpha_full.loc[:, alpha_columns]
    y_alpha = pd.to_numeric(
        alpha_fit[f"buy_return_{ENTRY_HORIZON}m_pct"],
        errors="coerce",
    )
    low = float(y_alpha.quantile(0.005))
    high = float(y_alpha.quantile(0.995))
    y_alpha_fit = y_alpha.clip(lower=low, upper=high)

    alpha_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20260919,
    )
    alpha_model.fit(x_alpha, y_alpha_fit)

    risk_fit = fit_period.loc[
        _label_eligible(fit_period, RISK_HORIZON)
        & pd.to_numeric(
            fit_period["buy_mae_15m_pct"], errors="coerce"
        ).notna()
    ].copy()
    risk_fit = _sample_training(risk_fit)
    x_risk_full = feature_frame(risk_fit)
    risk_columns = _usable_columns(x_risk_full)
    if not risk_columns:
        raise ValueError("no usable risk features")
    x_risk = x_risk_full.loc[:, risk_columns]
    y_risk = pd.to_numeric(
        risk_fit["buy_mae_15m_pct"], errors="coerce"
    ).le(SEVERE_MAE_PCT).astype(int)

    if y_risk.nunique() < 2:
        raise ValueError("severe-risk training labels contain fewer than two classes")

    risk_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20260920,
    )
    risk_model.fit(x_risk, y_risk)

    return EVRiskFold(
        alpha_model=alpha_model,
        risk_model=risk_model,
        alpha_columns=tuple(alpha_columns),
        risk_columns=tuple(risk_columns),
        train_target_low=low,
        train_target_high=high,
    )


def score_states(frame: pd.DataFrame, fold: EVRiskFold) -> pd.DataFrame:
    scored = frame.loc[_eligible(frame)].copy()
    x_alpha = feature_frame(scored).reindex(columns=fold.alpha_columns)
    x_risk = feature_frame(scored).reindex(columns=fold.risk_columns)

    predicted_gross = fold.alpha_model.predict(x_alpha)
    predicted_base = _modeled_net_from_predicted_gross(
        scored["c"],
        predicted_gross,
        _base_scenario(),
    )
    predicted_risk = fold.risk_model.predict_proba(x_risk)[:, 1]

    scored["predicted_mean_gross_10m_pct"] = predicted_gross
    scored["predicted_mean_base_10m_pct"] = predicted_base
    scored["predicted_severe_mae_prob"] = predicted_risk
    return scored


def _non_overlapping(
    scored: pd.DataFrame,
    *,
    ev_gate: float,
    risk_gate: bool,
) -> pd.DataFrame:
    mask = pd.to_numeric(
        scored["predicted_mean_base_10m_pct"], errors="coerce"
    ).ge(ev_gate)
    if risk_gate:
        mask &= pd.to_numeric(
            scored["predicted_severe_mae_prob"], errors="coerce"
        ).le(RISK_GATE_PROB)

    candidates = scored.loc[mask].copy()
    if candidates.empty:
        return candidates

    trades: list[dict[str, object]] = []
    for _, group in candidates.groupby(
        ["trading_day", "ticker"], sort=False
    ):
        group = group.sort_values("t", kind="stable")
        next_allowed_t = -1
        for row in group.itertuples(index=False):
            t = int(row.t)
            if t < next_allowed_t:
                continue
            entry = pd.to_numeric(
                pd.Series([getattr(row, "entry_price", np.nan)]),
                errors="coerce",
            ).iloc[0]
            realized_base = pd.to_numeric(
                pd.Series(
                    [
                        getattr(
                            row,
                            f"buy_return_{ENTRY_HORIZON}m_base_net_return_pct",
                            np.nan,
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]
            if pd.isna(entry) or float(entry) <= 0 or pd.isna(realized_base):
                continue
            trades.append(row._asdict())
            next_allowed_t = t + ENTRY_HORIZON * MINUTE_MS

    return pd.DataFrame(trades)


def _metrics(
    trades: pd.DataFrame,
    *,
    month: str,
    ev_gate: float,
    risk_gate: bool,
) -> dict[str, object]:
    gross_col = f"buy_return_{ENTRY_HORIZON}m_pct"
    base_col = f"buy_return_{ENTRY_HORIZON}m_base_net_return_pct"
    stress_col = f"buy_return_{ENTRY_HORIZON}m_stress_net_return_pct"

    if trades.empty:
        return {
            "month": month,
            "ev_gate_pct": ev_gate,
            "risk_gate": risk_gate,
            "trades": 0,
        }

    gross = pd.to_numeric(trades[gross_col], errors="coerce")
    base = pd.to_numeric(trades[base_col], errors="coerce")
    stress = pd.to_numeric(trades[stress_col], errors="coerce")
    mae = pd.to_numeric(trades["buy_mae_15m_pct"], errors="coerce")
    predicted_ev = pd.to_numeric(
        trades["predicted_mean_base_10m_pct"], errors="coerce"
    )
    predicted_risk = pd.to_numeric(
        trades["predicted_severe_mae_prob"], errors="coerce"
    )
    daily = trades.assign(_base=base).groupby("trading_day")["_base"].mean()

    return {
        "month": month,
        "ev_gate_pct": ev_gate,
        "risk_gate": risk_gate,
        "trades": int(len(trades)),
        "days": int(trades["trading_day"].nunique()),
        "trades_per_day": float(
            len(trades) / trades["trading_day"].nunique()
        ),
        "predicted_base_mean_pct": float(predicted_ev.mean()),
        "predicted_severe_mae_prob_mean": float(predicted_risk.mean()),
        "gross_mean_pct": float(gross.mean()),
        "base_mean_pct": float(base.mean()),
        "base_median_pct": float(base.median()),
        "base_positive_rate": float((base > 0).mean()),
        "base_p05_pct": float(base.quantile(0.05)),
        "stress_mean_pct": float(stress.mean()),
        "actual_severe_mae_rate": float(mae.le(SEVERE_MAE_PCT).mean()),
        "day_balanced_base_mean_pct": float(daily.mean()),
        "worst_day_mean_pct": float(daily.min()),
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )
        fold = train_fold(train)
        scored = score_states(holdout, fold)

        for ev_gate in EV_GATES_PCT:
            for use_risk_gate in (False, True):
                trades = _non_overlapping(
                    scored,
                    ev_gate=ev_gate,
                    risk_gate=use_risk_gate,
                )
                rows.append(
                    _metrics(
                        trades,
                        month=holdout_month,
                        ev_gate=ev_gate,
                        risk_gate=use_risk_gate,
                    )
                )
    return pd.DataFrame(rows)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (ev_gate, risk_gate), group in details.groupby(
        ["ev_gate_pct", "risk_gate"], sort=True
    ):
        valid = group.loc[group["trades"].ge(20)].copy()
        rows.append(
            {
                "ev_gate_pct": float(ev_gate),
                "risk_gate": bool(risk_gate),
                "months_tested": int(len(valid)),
                "min_trades": int(valid["trades"].min()) if len(valid) else 0,
                "median_trades_per_day": (
                    float(valid["trades_per_day"].median())
                    if len(valid)
                    else np.nan
                ),
                "gross_positive_months": int(
                    (valid["gross_mean_pct"] > 0).sum()
                ),
                "base_positive_months": int(
                    (valid["base_mean_pct"] > 0).sum()
                ),
                "day_positive_months": int(
                    (valid["day_balanced_base_mean_pct"] > 0).sum()
                ),
                "median_predicted_base_pct": (
                    float(valid["predicted_base_mean_pct"].median())
                    if len(valid)
                    else np.nan
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
                "median_actual_severe_mae_rate": (
                    float(valid["actual_severe_mae_rate"].median())
                    if len(valid)
                    else np.nan
                ),
                "median_base_p05_pct": (
                    float(valid["base_p05_pct"].median())
                    if len(valid)
                    else np.nan
                ),
                "median_stress_mean_pct": (
                    float(valid["stress_mean_pct"].median())
                    if len(valid)
                    else np.nan
                ),
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
            "=== MoneyMaker State EV + Risk Model v0.1 ===",
            "alpha=HistGradientBoostingRegressor(squared_error) on train-only 0.5%/99.5% winsorized 10m gross return",
            "risk=HistGradientBoostingClassifier for 15m MAE <= -5%",
            "validation=leave-one-month-out across already-seen January-March 2026",
            "score_information=current state only; predicted base friction uses current close, never next-minute open",
            "entry_diagnostics=predicted base EV gates 0.00%, 0.25%, 0.50%",
            "risk_diagnostic=optional predicted severe-MAE probability <=20%",
            "trade_accounting=10m non-overlapping per ticker-day; exact next-minute entry required only at execution",
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
        prog="python -m victory_trader.state_ev_risk_model"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path) for label, path in args.dataset
    }
    details = run_lomo(monthly)
    summary = summarize(details)
    report = render_report(details, summary)
    print(report)

    for path in (args.report, args.details_csv, args.summary_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
