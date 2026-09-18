from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from .candidate_v2_research import modeled_base_zero_return_cost_pct


HORIZONS = (5, 10, 15)
TOP_FRACTIONS = (0.01, 0.025, 0.05, 0.10)
EPISODE_KEYS = ["trading_day", "ticker"]
MAX_TRAIN_ROWS = 500_000

RAW_FEATURES = (
    "minutes_since_10pct_cross",
    "minutes_from_regular_open",
    "return_from_previous_close_pct",
    "running_hod_return_pct",
    "hod_distance_pct",
    "rebound_from_running_low_pct",
    "minutes_since_hod",
    "new_hod",
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
    "regular_vwap_distance_pct",
    "above_regular_vwap",
    "recent_deepest_pullback_15m_pct",
    "reclaim_prior_5m_high_after_pullback",
    "volume_accel_1m_vs_prior20m",
    "volume_accel_5m_vs_prior20m",
    "volume_3m_vs_postcross_peak",
    "transactions_1m",
    "transactions_5m",
    "dollar_volume_5m",
    "active_minute_fraction_15m",
    "runners_10pct_so_far",
    "runners_10pct_last_30m",
    "iwm_return_since_open_pct",
    "iwm_return_15m_pct",
    "iwm_volatility_15m_pct",
    "cross_rvol_cumulative_20d",
    "cross_rvol_5m_20d",
    "cross_premarket_return_pct",
    "cross_premarket_volume",
    "cross_premarket_high_return_pct",
    "cross_premarket_high_distance_pct",
    "cross_regular_open_gap_pct",
)

LOG1P_FEATURES = {
    "transactions_1m",
    "transactions_5m",
    "dollar_volume_5m",
    "runners_10pct_so_far",
    "runners_10pct_last_30m",
    "cross_premarket_volume",
}

BOOL_FEATURES = {
    "new_hod",
    "above_regular_vwap",
    "reclaim_prior_5m_high_after_pullback",
}


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"state panel missing model features: {sorted(missing)}")
    out = pd.DataFrame(index=frame.index)
    for feature in RAW_FEATURES:
        if feature in BOOL_FEATURES:
            values = frame[feature].astype("boolean").astype("Float64")
            out[feature] = pd.to_numeric(values, errors="coerce").astype(float)
        else:
            values = pd.to_numeric(frame[feature], errors="coerce").astype(float)
            if feature in LOG1P_FEATURES:
                values = np.log1p(values.clip(lower=0))
            out[feature] = values
    return out


def _deterministic_cap(frame: pd.DataFrame, max_rows: int = MAX_TRAIN_ROWS) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    positions = np.linspace(0, len(frame) - 1, max_rows, dtype=np.int64)
    return frame.iloc[positions].copy()


def _fit_model(train: pd.DataFrame, horizon: int) -> tuple[HistGradientBoostingRegressor, pd.DataFrame]:
    target_col = f"buy_return_{horizon}m_pct"
    target = pd.to_numeric(train[target_col], errors="coerce")
    valid = target.notna()
    fitted = train.loc[valid].sort_values(
        EPISODE_KEYS + ["t"], kind="stable"
    ).reset_index(drop=True)
    fitted = _deterministic_cap(fitted)
    y = pd.to_numeric(fitted[target_col], errors="coerce")

    lower = float(y.quantile(0.005))
    upper = float(y.quantile(0.995))
    y_fit = y.clip(lower=lower, upper=upper)

    x = prepare_features(fitted)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=20260918,
    )
    model.fit(x, y_fit)
    return model, fitted


def _score(frame: pd.DataFrame, model: HistGradientBoostingRegressor) -> pd.DataFrame:
    result = frame.copy()
    x = prepare_features(result)
    result["predicted_gross_pct"] = model.predict(x)
    current_close = pd.to_numeric(result["c"], errors="coerce")
    result["modeled_base_cost_from_current_close_pct"] = current_close.apply(
        modeled_base_zero_return_cost_pct
    )
    result["predicted_after_cost_score_pct"] = (
        result["predicted_gross_pct"]
        - result["modeled_base_cost_from_current_close_pct"]
    )
    return result


def _first_signal_per_episode(
    scored: pd.DataFrame,
    score_cut: float,
) -> pd.DataFrame:
    candidates = scored.loc[
        pd.to_numeric(
            scored["predicted_after_cost_score_pct"], errors="coerce"
        ).ge(score_cut)
    ].copy()
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values(EPISODE_KEYS + ["t"], kind="stable")
        .drop_duplicates(EPISODE_KEYS, keep="first")
        .reset_index(drop=True)
    )


def _evaluate_signals(
    signals: pd.DataFrame,
    *,
    month: str,
    horizon: int,
    top_fraction: float,
    score_cut: float,
    all_scored: pd.DataFrame,
) -> dict[str, object]:
    gross_col = f"buy_return_{horizon}m_pct"
    base_col = f"buy_return_{horizon}m_base_net_return_pct"
    stress_col = f"buy_return_{horizon}m_stress_net_return_pct"

    entry = pd.to_numeric(signals["entry_price"], errors="coerce")
    executable = entry.notna()
    gross = pd.to_numeric(signals[gross_col], errors="coerce")
    observed = executable & gross.notna()

    base = pd.to_numeric(signals[base_col], errors="coerce")
    stress = pd.to_numeric(signals[stress_col], errors="coerce")

    scored_target = pd.to_numeric(all_scored[gross_col], errors="coerce")
    scored_pred = pd.to_numeric(
        all_scored["predicted_after_cost_score_pct"], errors="coerce"
    )
    corr_mask = scored_target.notna() & scored_pred.notna()
    if int(corr_mask.sum()) >= 10:
        rho = float(
            spearmanr(
                scored_pred.loc[corr_mask],
                scored_target.loc[corr_mask],
            ).statistic
        )
    else:
        rho = np.nan

    day_base = (
        pd.DataFrame(
            {
                "trading_day": signals.loc[observed, "trading_day"].astype(str),
                "base": base.loc[observed],
            }
        )
        .groupby("trading_day")["base"]
        .mean()
        if observed.any()
        else pd.Series(dtype=float)
    )

    return {
        "month": month,
        "horizon_min": horizon,
        "top_fraction": top_fraction,
        "score_cut": score_cut,
        "state_rows_scored": int(len(all_scored)),
        "episodes": int(all_scored[EPISODE_KEYS].drop_duplicates().shape[0]),
        "signal_n": int(len(signals)),
        "executable_n": int(executable.sum()),
        "observed_n": int(observed.sum()),
        "signal_episode_rate": (
            float(len(signals) / all_scored[EPISODE_KEYS].drop_duplicates().shape[0])
            if len(all_scored)
            else np.nan
        ),
        "executable_rate": (
            float(executable.mean()) if len(signals) else np.nan
        ),
        "observed_given_executable_rate": (
            float(observed.sum() / executable.sum())
            if executable.any()
            else np.nan
        ),
        "score_target_spearman": rho,
        "predicted_gross_mean_pct": (
            float(signals.loc[observed, "predicted_gross_pct"].mean())
            if observed.any()
            else np.nan
        ),
        "predicted_after_cost_mean_pct": (
            float(
                signals.loc[
                    observed, "predicted_after_cost_score_pct"
                ].mean()
            )
            if observed.any()
            else np.nan
        ),
        "gross_mean_pct": (
            float(gross.loc[observed].mean()) if observed.any() else np.nan
        ),
        "gross_median_pct": (
            float(gross.loc[observed].median()) if observed.any() else np.nan
        ),
        "base_mean_pct": (
            float(base.loc[observed].mean()) if observed.any() else np.nan
        ),
        "base_median_pct": (
            float(base.loc[observed].median()) if observed.any() else np.nan
        ),
        "base_positive_rate": (
            float((base.loc[observed] > 0).mean())
            if observed.any()
            else np.nan
        ),
        "base_p05_pct": (
            float(base.loc[observed].quantile(0.05))
            if observed.any()
            else np.nan
        ),
        "stress_mean_pct": (
            float(stress.loc[observed].mean()) if observed.any() else np.nan
        ),
        "day_balanced_base_mean_pct": (
            float(day_base.mean()) if not day_base.empty else np.nan
        ),
        "days_with_observed_signal": int(day_base.size),
    }


def run_leave_one_month_out(
    monthly_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if len(monthly_frames) < 3:
        raise ValueError("state action model requires at least three development months")

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

        for horizon in HORIZONS:
            model, fitted = _fit_model(train, horizon)
            train_scored = _score(fitted, model)
            test_scored = _score(holdout, model)

            train_scores = pd.to_numeric(
                train_scored["predicted_after_cost_score_pct"],
                errors="coerce",
            ).dropna()

            for top_fraction in TOP_FRACTIONS:
                score_cut = float(train_scores.quantile(1.0 - top_fraction))
                signals = _first_signal_per_episode(test_scored, score_cut)
                rows.append(
                    _evaluate_signals(
                        signals,
                        month=holdout_month,
                        horizon=horizon,
                        top_fraction=top_fraction,
                        score_cut=score_cut,
                        all_scored=test_scored,
                    )
                )
    return pd.DataFrame(rows)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon, fraction), group in details.groupby(
        ["horizon_min", "top_fraction"], sort=True
    ):
        valid = group.loc[group["observed_n"].ge(30)].copy()
        rows.append(
            {
                "horizon_min": int(horizon),
                "top_fraction": float(fraction),
                "months_tested": int(len(valid)),
                "min_observed_n": (
                    int(valid["observed_n"].min()) if len(valid) else 0
                ),
                "median_signal_episode_rate": (
                    float(valid["signal_episode_rate"].median())
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
                "median_spearman": (
                    float(valid["score_target_spearman"].median())
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
                "median_day_balanced_base_mean_pct": (
                    float(valid["day_balanced_base_mean_pct"].median())
                    if len(valid)
                    else np.nan
                ),
                "median_base_positive_rate": (
                    float(valid["base_positive_rate"].median())
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
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        [
            "base_positive_months",
            "day_balanced_base_positive_months",
            "median_base_mean_pct",
            "worst_base_mean_pct",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def render_report(details: pd.DataFrame, summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State Action-Value Model v0.1 ===",
            "method=leave-one-month-out across January-March 2026",
            "model=HistGradientBoostingRegressor trained on gross forward return",
            "execution=prediction ranked by predicted gross minus current-close modeled base friction",
            "selection_cut=learned on training months only; test month does not set its own percentile",
            "duplicate_control=first qualifying state per ticker-day episode",
            "target_winsorization=train-only 0.5%/99.5%; evaluation uses raw held-out returns",
            "fresh_months_consumed=none",
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
        prog="python -m victory_trader.state_action_model"
    )
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    frames = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    details = run_leave_one_month_out(frames)
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
