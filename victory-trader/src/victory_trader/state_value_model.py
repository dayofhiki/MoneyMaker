from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .execution_costs import DEFAULT_EXECUTION_SCENARIOS


MINUTE_MS = 60_000
HORIZONS = (5, 10, 15)
SELECTION_QUANTILES = (0.95, 0.98, 0.99)
TRAIN_MIN_ACTIVE_FRACTION = 0.80
TRAIN_SAMPLE_EVERY_MINUTES = 3

RAW_FEATURES = (
    "return_from_previous_close_pct",
    "minutes_since_10pct_cross",
    "minutes_from_regular_open",
    "regular_vwap_distance_pct",
    "running_hod_return_pct",
    "hod_distance_pct",
    "rebound_from_running_low_pct",
    "minutes_since_hod",
    "trailing_return_1m_pct",
    "trailing_return_3m_pct",
    "trailing_return_5m_pct",
    "trailing_return_10m_pct",
    "trailing_return_15m_pct",
    "volatility_5m_pct",
    "volatility_15m_pct",
    "bar_range_pct",
    "bar_body_pct",
    "bar_close_location",
    "range_contraction_3m_vs_15m",
    "volume_accel_1m_vs_prior20m",
    "volume_accel_5m_vs_prior20m",
    "volume_3m_vs_postcross_peak",
    "active_minute_fraction_15m",
    "recent_deepest_pullback_15m_pct",
    "runners_10pct_so_far",
    "runners_10pct_last_30m",
    "iwm_return_since_open_pct",
    "iwm_return_15m_pct",
    "iwm_volatility_15m_pct",
    "cross_rvol_cumulative_20d",
    "cross_rvol_5m_20d",
    "cross_premarket_return_pct",
    "cross_premarket_high_return_pct",
    "cross_premarket_high_distance_pct",
    "cross_regular_open_gap_pct",
)

BOOLEAN_FEATURES = (
    "above_regular_vwap",
    "new_hod",
    "reclaim_prior_5m_high_after_pullback",
)

LOG_FEATURES = (
    "volume_1m",
    "volume_3m",
    "volume_5m",
    "transactions_1m",
    "transactions_5m",
    "dollar_volume_5m",
    "cross_premarket_volume",
)


@dataclass(frozen=True)
class FoldModel:
    horizon: int
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    calibration_cutoffs: dict[float, float]


def _base_scenario():
    return next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == "base")


def _modeled_net_from_predicted_gross(
    entry_price: pd.Series,
    predicted_gross_pct: np.ndarray,
    scenario,
) -> np.ndarray:
    entry = pd.to_numeric(entry_price, errors="coerce").to_numpy(dtype=float)
    gross = np.asarray(predicted_gross_pct, dtype=float)
    result = np.full(len(entry), np.nan, dtype=float)

    valid = np.isfinite(entry) & np.isfinite(gross) & (entry > 0)
    if not valid.any():
        return result

    e = entry[valid]
    raw_exit = e * (1.0 + gross[valid] / 100.0)
    exit_valid = raw_exit > 0

    entry_half = np.maximum(
        e * scenario.half_spread_bps / 10_000.0,
        scenario.min_half_spread_cents / 100.0,
    )
    buy = e * (1.0 + scenario.slippage_bps / 10_000.0) + entry_half

    exit_half = np.maximum(
        raw_exit * scenario.half_spread_bps / 10_000.0,
        scenario.min_half_spread_cents / 100.0,
    )
    sell = np.maximum(
        raw_exit * (1.0 - scenario.slippage_bps / 10_000.0) - exit_half,
        0.0,
    )
    sell *= 1.0 - scenario.sell_fee_bps / 10_000.0

    values = np.full(len(e), -100.0, dtype=float)
    values[exit_valid] = (sell[exit_valid] / buy[exit_valid] - 1.0) * 100.0
    result[np.flatnonzero(valid)] = values
    return result


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}

    for column in RAW_FEATURES:
        if column in frame.columns:
            data[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            data[column] = pd.Series(np.nan, index=frame.index, dtype=float)

    for column in BOOLEAN_FEATURES:
        if column in frame.columns:
            data[column] = (
                frame[column]
                .astype("boolean")
                .astype("Float64")
                .astype(float)
            )
        else:
            data[column] = pd.Series(np.nan, index=frame.index, dtype=float)

    for column in LOG_FEATURES:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").clip(lower=0)
            data[f"log1p_{column}"] = np.log1p(values)
        else:
            data[f"log1p_{column}"] = pd.Series(
                np.nan, index=frame.index, dtype=float
            )

    return pd.DataFrame(data, index=frame.index)


def _eligible(frame: pd.DataFrame, horizon: int | None = None) -> pd.Series:
    """State eligibility using only information known at decision time.

    The horizon argument is accepted for backward compatibility but deliberately
    does not inspect future return labels or the next-minute entry price.
    """
    return (
        pd.to_numeric(frame["c"], errors="coerce").notna()
        & pd.to_numeric(
            frame["active_minute_fraction_15m"], errors="coerce"
        ).ge(TRAIN_MIN_ACTIVE_FRACTION)
    )


def _label_eligible(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Training-label availability. Never use this mask to decide live signals."""
    return (
        _eligible(frame)
        & pd.to_numeric(
            frame[f"buy_return_{horizon}m_pct"], errors="coerce"
        ).notna()
    )


def _chronological_fit_calibration_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = sorted(frame["trading_day"].astype(str).unique())
    if len(days) < 10:
        cut = max(1, int(len(days) * 0.8))
    else:
        cut = max(1, int(len(days) * 0.80))
    cut = min(cut, len(days) - 1)
    fit_days = set(days[:cut])
    fit = frame.loc[frame["trading_day"].astype(str).isin(fit_days)].copy()
    calibration = frame.loc[
        ~frame["trading_day"].astype(str).isin(fit_days)
    ].copy()
    return fit, calibration


def _sample_training_states(frame: pd.DataFrame) -> pd.DataFrame:
    minutes = pd.to_numeric(
        frame["minutes_since_10pct_cross"], errors="coerce"
    )
    return frame.loc[
        minutes.mod(TRAIN_SAMPLE_EVERY_MINUTES).eq(0)
    ].copy()


def train_fold_model(
    train: pd.DataFrame,
    horizon: int,
) -> FoldModel:
    scoreable = train.loc[_eligible(train)].copy()
    fit_period, calibration = _chronological_fit_calibration_split(scoreable)
    fit = fit_period.loc[_label_eligible(fit_period, horizon)].copy()
    fit = _sample_training_states(fit)

    X_fit_full = feature_frame(fit)
    usable_columns = [
        column
        for column in X_fit_full.columns
        if X_fit_full[column].notna().any()
    ]
    if not usable_columns:
        raise ValueError("no usable state features in training fold")
    X_fit = X_fit_full.loc[:, usable_columns]
    y_fit = pd.to_numeric(
        fit[f"buy_return_{horizon}m_pct"], errors="coerce"
    ).to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.06,
        max_iter=140,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=1.0,
        random_state=20260918 + horizon,
    )
    model.fit(X_fit, y_fit)

    X_cal = feature_frame(calibration).reindex(columns=usable_columns)
    pred_gross = model.predict(X_cal)
    pred_base = _modeled_net_from_predicted_gross(
        calibration["c"],
        pred_gross,
        _base_scenario(),
    )
    calibration_cutoffs = {
        quantile: float(np.nanquantile(pred_base, quantile))
        for quantile in SELECTION_QUANTILES
    }

    return FoldModel(
        horizon=horizon,
        model=model,
        feature_columns=tuple(usable_columns),
        calibration_cutoffs=calibration_cutoffs,
    )


def _simulate_non_overlapping_trades(
    scored: pd.DataFrame,
    *,
    horizon: int,
    cutoff: float,
) -> pd.DataFrame:
    selected = scored.loc[
        pd.to_numeric(scored["predicted_base_net_pct"], errors="coerce").ge(
            cutoff
        )
    ].copy()

    if selected.empty:
        return selected

    trades = []
    for _, group in selected.groupby(["trading_day", "ticker"], sort=False):
        group = group.sort_values("t", kind="stable")
        last_exit_t = -1
        for row in group.itertuples(index=False):
            state_t = int(row.t)
            if state_t < last_exit_t:
                continue
            entry_price = pd.to_numeric(
                pd.Series([getattr(row, "entry_price", np.nan)]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(entry_price) or float(entry_price) <= 0:
                continue
            trades.append(row._asdict())
            last_exit_t = state_t + horizon * MINUTE_MS

    if not trades:
        return selected.iloc[0:0].copy()
    return pd.DataFrame(trades)


def _metrics(
    trades: pd.DataFrame,
    *,
    month: str,
    horizon: int,
    quantile: float,
    cutoff: float,
) -> dict[str, object]:
    gross_col = f"buy_return_{horizon}m_pct"
    base_col = f"buy_return_{horizon}m_base_net_return_pct"
    stress_col = f"buy_return_{horizon}m_stress_net_return_pct"

    gross = pd.to_numeric(trades[gross_col], errors="coerce")
    base = pd.to_numeric(trades[base_col], errors="coerce")
    stress = pd.to_numeric(trades[stress_col], errors="coerce")
    pred = pd.to_numeric(trades["predicted_base_net_pct"], errors="coerce")

    daily = (
        trades.assign(_base=base)
        .groupby("trading_day")["_base"]
        .mean()
        if not trades.empty
        else pd.Series(dtype=float)
    )

    return {
        "month": month,
        "horizon_min": horizon,
        "selection_quantile": quantile,
        "predicted_base_cutoff_pct": cutoff,
        "trades": int(len(trades)),
        "ticker_day_episodes": int(
            trades[["trading_day", "ticker"]].drop_duplicates().shape[0]
        )
        if not trades.empty
        else 0,
        "days": int(trades["trading_day"].nunique()) if not trades.empty else 0,
        "trades_per_day": (
            float(len(trades) / trades["trading_day"].nunique())
            if not trades.empty and trades["trading_day"].nunique()
            else np.nan
        ),
        "predicted_base_mean_pct": float(pred.mean())
        if not trades.empty
        else np.nan,
        "gross_mean_pct": float(gross.mean()) if not trades.empty else np.nan,
        "base_mean_pct": float(base.mean()) if not trades.empty else np.nan,
        "base_median_pct": float(base.median()) if not trades.empty else np.nan,
        "base_positive_rate": float((base > 0).mean())
        if not trades.empty
        else np.nan,
        "base_p05_pct": float(base.quantile(0.05))
        if not trades.empty
        else np.nan,
        "stress_mean_pct": float(stress.mean())
        if not trades.empty
        else np.nan,
        "day_balanced_base_mean_pct": float(daily.mean())
        if not daily.empty
        else np.nan,
        "worst_day_mean_pct": float(daily.min())
        if not daily.empty
        else np.nan,
    }


def run_lomo(monthly_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scored_samples = []

    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )

        for horizon in HORIZONS:
            fold = train_fold_model(train, horizon)

            eligible = holdout.loc[_eligible(holdout)].copy()
            X_holdout = feature_frame(eligible).reindex(
                columns=fold.feature_columns
            )
            pred_gross = fold.model.predict(X_holdout)
            pred_base = _modeled_net_from_predicted_gross(
                eligible["c"],
                pred_gross,
                _base_scenario(),
            )
            eligible["predicted_gross_pct"] = pred_gross
            eligible["predicted_base_net_pct"] = pred_base

            sample = (
                eligible.nlargest(
                    min(2000, len(eligible)),
                    "predicted_base_net_pct",
                )[
                    [
                        "trading_day",
                        "ticker",
                        "t",
                        "entry_price",
                        "predicted_gross_pct",
                        "predicted_base_net_pct",
                        f"buy_return_{horizon}m_pct",
                        f"buy_return_{horizon}m_base_net_return_pct",
                    ]
                ]
                .copy()
            )
            sample["month"] = holdout_month
            sample["horizon_min"] = horizon
            scored_samples.append(sample)

            for quantile, cutoff in fold.calibration_cutoffs.items():
                trades = _simulate_non_overlapping_trades(
                    eligible,
                    horizon=horizon,
                    cutoff=cutoff,
                )
                rows.append(
                    _metrics(
                        trades,
                        month=holdout_month,
                        horizon=horizon,
                        quantile=quantile,
                        cutoff=cutoff,
                    )
                )

    return pd.DataFrame(rows), pd.concat(scored_samples, ignore_index=True)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon, quantile), group in details.groupby(
        ["horizon_min", "selection_quantile"], sort=True
    ):
        valid = group.loc[group["trades"].ge(20)].copy()
        rows.append(
            {
                "horizon_min": int(horizon),
                "selection_quantile": float(quantile),
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
                "median_stress_mean_pct": float(
                    valid["stress_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_base_p05_pct": float(
                    valid["base_p05_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_worst_day_mean_pct": float(
                    valid["worst_day_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        [
            "base_positive_months",
            "day_positive_months",
            "median_base_mean_pct",
            "worst_base_mean_pct",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def render_report(details: pd.DataFrame, stability: pd.DataFrame) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State Action-Value Model v0.2 ===",
            "model=HistGradientBoostingRegressor(loss=absolute_error)",
            "features=point-in-time price/path/volume/liquidity/market-context state variables only",
            "target=gross future return; predicted base-net computed from current close and base execution model (no next-minute-price lookahead)",
            "validation=leave-one-month-out across already-seen January-March 2026",
            "calibration=chronological final 20% of the two training months",
            "training_density=every third post-cross minute to reduce serial duplication",
            "policy_eval=score every held-out minute without conditioning on future label availability; one open position per ticker; re-entry allowed after fixed horizon exit",
            "selection=top 5%, 2%, 1% predicted base-net based on training calibration only",
            "NOTE=this is development research, not an external proof or deployable policy",
            "",
            "=== Cross-month stability ===",
            stability.to_string(index=False),
            "",
            "=== Month-by-month policy diagnostics ===",
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
        prog="python -m victory_trader.state_value_model"
    )
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--stability-csv", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    details, scores = run_lomo(monthly)
    stability = summarize(details)
    report = render_report(details, stability)
    print(report)

    for path in (
        args.report,
        args.details_csv,
        args.stability_csv,
        args.scores_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    stability.to_csv(args.stability_csv, index=False)
    scores.to_csv(args.scores_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
