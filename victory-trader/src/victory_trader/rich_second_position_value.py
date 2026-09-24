"""Request 171: richer one-second POSITION microstructure proxies.

No new dates. Reuse Request-166 BUY-gated POSITION rows and enrich each causal
state from the still-accessible historical one-second aggregate feed. Historical
tick trades/NBBO are not used.

The intervention changes only the information set of the surviving
multi-minute excess remaining-option regressor. The one-minute HOLD target,
entry policy, execution costs, and 30-minute safety cap are untouched.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import load_settings
from .massive_client import MassiveClient
from .second_path_attention_probe import (
    SECOND_CACHE_DIR,
    _second_frame,
    second_feature_window,
)
from .selected_hot_position_value_observability import (
    MODEL_FEATURES,
    OPTION_SEED,
    apply_excess_target,
    fit_minute_baselines,
    predict_option,
    train_option_model,
)
from .buy_gated_transition_option_value import evaluate_option_signal

REQUEST_ID = 171
MINUTE_MS = 60_000

RICH_SECOND_FEATURES = (
    "sec_last5_return_pct",
    "sec_prev5_return_pct",
    "sec_accel_5_pct",
    "sec_last15_return_pct",
    "sec_prev15_return_pct",
    "sec_accel_15_pct",
    "sec_volume_burst_5",
    "sec_transactions_burst_5",
    "sec_volume_last5_share",
    "sec_transactions_last5_share",
    "sec_signed_volume_imbalance_60",
    "sec_signed_transactions_imbalance_60",
    "sec_return_efficiency_60",
    "sec_sign_flip_rate_60",
    "sec_seconds_since_high",
    "sec_seconds_since_low",
    "sec_close_location_60",
    "sec_close_vs_vwap_pct",
    "sec_volatility_ratio_10_to_prev50",
    "sec_mean_trade_size_ratio_10_to_prev50",
)

MIN_EVAL_FEATURE_COVERAGE = 0.85
MIN_POOLED_SPEARMAN_IMPROVEMENT = 0.03
MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT = 0.02
MIN_SELECTED_RATE = 0.10
MIN_GOOD_DAYS = 4


@dataclass(frozen=True)
class RichSecondOptionModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def _safe_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _window_return(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    first_open = pd.to_numeric(
        pd.Series([frame.iloc[0].get("o")]), errors="coerce"
    ).iloc[0]
    last_close = pd.to_numeric(
        pd.Series([frame.iloc[-1].get("c")]), errors="coerce"
    ).iloc[0]
    if (
        pd.isna(first_open)
        or pd.isna(last_close)
        or float(first_open) <= 0
        or float(last_close) <= 0
    ):
        return np.nan
    return float((float(last_close) / float(first_open) - 1.0) * 100.0)


def rich_second_features(
    seconds: pd.DataFrame,
    state_t: int,
) -> dict[str, float]:
    window = second_feature_window(seconds, state_t)
    empty = {name: np.nan for name in RICH_SECOND_FEATURES}
    if window.empty:
        return empty

    t = pd.to_numeric(window["t"], errors="coerce")
    close = pd.to_numeric(window["c"], errors="coerce")
    volume = pd.to_numeric(window.get("v"), errors="coerce").fillna(0.0)
    transactions = pd.to_numeric(
        window.get("n"), errors="coerce"
    ).fillna(0.0)

    last5 = window.loc[t.ge(state_t - 5_000)]
    prev5 = window.loc[
        t.ge(state_t - 10_000) & t.lt(state_t - 5_000)
    ]
    last15 = window.loc[t.ge(state_t - 15_000)]
    prev15 = window.loc[
        t.ge(state_t - 30_000) & t.lt(state_t - 15_000)
    ]
    last10_mask = t.ge(state_t - 10_000)
    prev50_mask = t.lt(state_t - 10_000)

    last5_return = _window_return(last5)
    prev5_return = _window_return(prev5)
    last15_return = _window_return(last15)
    prev15_return = _window_return(prev15)

    volume_total = float(volume.sum())
    tx_total = float(transactions.sum())
    v5 = _safe_sum(last5.get("v", pd.Series(dtype=float)))
    pv5 = _safe_sum(prev5.get("v", pd.Series(dtype=float)))
    n5 = _safe_sum(last5.get("n", pd.Series(dtype=float)))
    pn5 = _safe_sum(prev5.get("n", pd.Series(dtype=float)))

    log_returns = np.log(close.where(close.gt(0))).diff()
    signs = np.sign(log_returns.fillna(0.0).to_numpy(dtype=float))
    last_nonzero = 0.0
    for i in range(len(signs)):
        if signs[i] == 0.0:
            signs[i] = last_nonzero
        else:
            last_nonzero = signs[i]

    signed_volume = float(np.sum(signs * volume.to_numpy(dtype=float)))
    signed_tx = float(
        np.sum(signs * transactions.to_numpy(dtype=float))
    )

    valid_returns = log_returns.dropna().to_numpy(dtype=float)
    net_log = (
        float(np.log(close.dropna().iloc[-1] / close.dropna().iloc[0]))
        if close.dropna().shape[0] >= 2
        and float(close.dropna().iloc[0]) > 0
        and float(close.dropna().iloc[-1]) > 0
        else np.nan
    )
    path_abs = (
        float(np.sum(np.abs(valid_returns)))
        if len(valid_returns)
        else np.nan
    )
    efficiency = (
        float(net_log / path_abs)
        if np.isfinite(net_log)
        and np.isfinite(path_abs)
        and path_abs > 0
        else np.nan
    )

    nz = signs[signs != 0]
    flip_rate = (
        float(np.mean(nz[1:] != nz[:-1]))
        if len(nz) >= 2
        else np.nan
    )

    valid_close = close.dropna()
    if len(valid_close):
        high_idx = int(valid_close.idxmax())
        low_idx = int(valid_close.idxmin())
        high_t = pd.to_numeric(
            pd.Series([window.loc[high_idx, "t"]]), errors="coerce"
        ).iloc[0]
        low_t = pd.to_numeric(
            pd.Series([window.loc[low_idx, "t"]]), errors="coerce"
        ).iloc[0]
        seconds_since_high = (
            float(max(state_t - 1_000 - int(high_t), 0) / 1_000.0)
            if pd.notna(high_t)
            else np.nan
        )
        seconds_since_low = (
            float(max(state_t - 1_000 - int(low_t), 0) / 1_000.0)
            if pd.notna(low_t)
            else np.nan
        )
        low_close = float(valid_close.min())
        high_close = float(valid_close.max())
        last_close = float(valid_close.iloc[-1])
        close_location = (
            float((last_close - low_close) / (high_close - low_close))
            if high_close > low_close
            else 0.5
        )
    else:
        seconds_since_high = np.nan
        seconds_since_low = np.nan
        close_location = np.nan
        last_close = np.nan

    vwap = (
        float(
            np.sum(close.fillna(0.0).to_numpy(dtype=float)
                   * volume.to_numpy(dtype=float))
            / volume_total
        )
        if volume_total > 0
        else np.nan
    )
    close_vs_vwap = (
        float((last_close / vwap - 1.0) * 100.0)
        if np.isfinite(last_close)
        and np.isfinite(vwap)
        and vwap > 0
        else np.nan
    )

    def _vol(mask: pd.Series) -> float:
        c = close.loc[mask]
        r = np.log(c.where(c.gt(0))).diff().dropna()
        return float(r.std(ddof=0)) if len(r) >= 2 else np.nan

    vol10 = _vol(last10_mask)
    vol50 = _vol(prev50_mask)
    vol_ratio = (
        float(vol10 / vol50)
        if np.isfinite(vol10) and np.isfinite(vol50) and vol50 > 0
        else np.nan
    )

    v10 = float(volume.loc[last10_mask].sum())
    n10 = float(transactions.loc[last10_mask].sum())
    v50 = float(volume.loc[prev50_mask].sum())
    n50 = float(transactions.loc[prev50_mask].sum())
    mean_size10 = v10 / n10 if n10 > 0 else np.nan
    mean_size50 = v50 / n50 if n50 > 0 else np.nan
    size_ratio = (
        float(mean_size10 / mean_size50)
        if np.isfinite(mean_size10)
        and np.isfinite(mean_size50)
        and mean_size50 > 0
        else np.nan
    )

    return {
        "sec_last5_return_pct": last5_return,
        "sec_prev5_return_pct": prev5_return,
        "sec_accel_5_pct": (
            last5_return - prev5_return
            if np.isfinite(last5_return) and np.isfinite(prev5_return)
            else np.nan
        ),
        "sec_last15_return_pct": last15_return,
        "sec_prev15_return_pct": prev15_return,
        "sec_accel_15_pct": (
            last15_return - prev15_return
            if np.isfinite(last15_return) and np.isfinite(prev15_return)
            else np.nan
        ),
        "sec_volume_burst_5": v5 / pv5 if pv5 > 0 else np.nan,
        "sec_transactions_burst_5": n5 / pn5 if pn5 > 0 else np.nan,
        "sec_volume_last5_share": (
            v5 / volume_total if volume_total > 0 else np.nan
        ),
        "sec_transactions_last5_share": (
            n5 / tx_total if tx_total > 0 else np.nan
        ),
        "sec_signed_volume_imbalance_60": (
            signed_volume / volume_total if volume_total > 0 else np.nan
        ),
        "sec_signed_transactions_imbalance_60": (
            signed_tx / tx_total if tx_total > 0 else np.nan
        ),
        "sec_return_efficiency_60": efficiency,
        "sec_sign_flip_rate_60": flip_rate,
        "sec_seconds_since_high": seconds_since_high,
        "sec_seconds_since_low": seconds_since_low,
        "sec_close_location_60": close_location,
        "sec_close_vs_vwap_pct": close_vs_vwap,
        "sec_volatility_ratio_10_to_prev50": vol_ratio,
        "sec_mean_trade_size_ratio_10_to_prev50": size_ratio,
    }


def enrich_phase(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
) -> int:
    frame = pd.read_parquet(input_path).copy()
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )

    cache: dict[tuple[str, str], pd.DataFrame] = {}
    records: list[dict[str, float]] = []
    for row in frame.loc[
        :, ["trading_day", "ticker", "state_t"]
    ].to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        key = (day, ticker)
        if key not in cache:
            payload = client.second_bars_range(
                ticker,
                date.fromisoformat(day),
                date.fromisoformat(day),
                adjusted=False,
            )
            cache[key] = _second_frame(payload)
        records.append(
            rich_second_features(cache[key], int(row["state_t"]))
        )

    enriched = pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(records)],
        axis=1,
    )
    coverage = {
        name: float(
            pd.to_numeric(enriched[name], errors="coerce").notna().mean()
        )
        for name in RICH_SECOND_FEATURES
    }
    core_coverage = float(
        enriched.loc[:, RICH_SECOND_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False, compression="zstd")
    audit = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "rows": int(len(enriched)),
        "ticker_days": int(len(cache)),
        "any_rich_second_feature_coverage": core_coverage,
        "feature_coverage": coverage,
        "client_stats": client.stats.to_dict(),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_rich_option(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> RichSecondOptionModel:
    columns = tuple(
        dict.fromkeys([*MODEL_FEATURES, *RICH_SECOND_FEATURES])
    )
    target = pd.to_numeric(
        fit["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError("request 171 has insufficient fit support")

    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=OPTION_SEED,
    )
    model.fit(
        _feature_frame(train, columns),
        y.clip(lower=low, upper=high),
    )

    cal_target = pd.to_numeric(
        calibration["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    if int(cal_valid.sum()) < 500:
        raise ValueError("request 171 has insufficient calibration support")
    raw = model.predict(
        _feature_frame(calibration.loc[cal_valid], columns)
    )
    offset = float(
        cal_target.loc[cal_valid].mean() - float(np.mean(raw))
    )
    return RichSecondOptionModel(
        model=model,
        feature_columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_rich_option(
    frame: pd.DataFrame,
    fitted: RichSecondOptionModel,
) -> np.ndarray:
    return (
        fitted.model.predict(
            _feature_frame(frame, fitted.feature_columns)
        )
        + fitted.offset
    )


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    evaluation_audit_path: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    evaluation = pd.read_parquet(evaluation_path)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    baseline_model = train_option_model(fit, calibration)
    rich_model = train_rich_option(fit, calibration)

    scored = evaluation.copy()
    scored["baseline_option_score"] = predict_option(
        scored, baseline_model
    )
    scored["rich_second_option_score"] = predict_rich_option(
        scored, rich_model
    )

    baseline = evaluate_option_signal(
        scored,
        "baseline_option_score",
        label="request166_baseline_option",
    )
    candidate = evaluate_option_signal(
        scored,
        "rich_second_option_score",
        label="rich_second_option",
    )

    pooled_improvement = (
        None
        if baseline["global_spearman"] is None
        or candidate["global_spearman"] is None
        else float(
            candidate["global_spearman"]
            - baseline["global_spearman"]
        )
    )
    minute_improvement = (
        None
        if baseline["same_minute_spearman_median"] is None
        or candidate["same_minute_spearman_median"] is None
        else float(
            candidate["same_minute_spearman_median"]
            - baseline["same_minute_spearman_median"]
        )
    )
    eval_audit = json.loads(
        evaluation_audit_path.read_text(encoding="utf-8")
    )
    feature_coverage = float(
        eval_audit["any_rich_second_feature_coverage"]
    )

    promotion = bool(
        feature_coverage >= MIN_EVAL_FEATURE_COVERAGE
        and pooled_improvement is not None
        and pooled_improvement >= MIN_POOLED_SPEARMAN_IMPROVEMENT
        and minute_improvement is not None
        and minute_improvement
        >= MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT
        and candidate["selected_rate"] is not None
        and float(candidate["selected_rate"]) >= MIN_SELECTED_RATE
        and candidate["selected_day_balanced_excess_mean_pct"]
        is not None
        and float(
            candidate["selected_day_balanced_excess_mean_pct"]
        )
        > 0
        and candidate["selected_day_bootstrap"]["ci_low_pct"]
        is not None
        and float(
            candidate["selected_day_bootstrap"]["ci_low_pct"]
        )
        > 0
        and candidate["positive_spearman_and_excess_days"]
        >= MIN_GOOD_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "tick_trade_data_used": False,
        "nbbo_data_used": False,
        "one_minute_hold_model_changed": False,
        "remaining_option_target_changed": False,
        "rich_second_features": list(RICH_SECOND_FEATURES),
        "evaluation_feature_coverage": feature_coverage,
        "baseline": baseline,
        "candidate": candidate,
        "pooled_spearman_improvement": pooled_improvement,
        "same_minute_median_improvement": minute_improvement,
        "model_diagnostics": {
            "feature_count": len(rich_model.feature_columns),
            "offset": rich_model.offset,
            "winsor_low_pct": rich_model.winsor_low,
            "winsor_high_pct": rich_model.winsor_high,
        },
        "frozen_gate": {
            "min_eval_feature_coverage": MIN_EVAL_FEATURE_COVERAGE,
            "min_pooled_spearman_improvement": (
                MIN_POOLED_SPEARMAN_IMPROVEMENT
            ),
            "min_same_minute_median_improvement": (
                MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT
            ),
            "min_selected_rate": MIN_SELECTED_RATE,
            "min_good_days": MIN_GOOD_DAYS,
            "bootstrap_ci_low_must_be_positive": True,
        },
        "promotion_gate_pass": promotion,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored.to_parquet(
        scored_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    enrich = sub.add_parser("enrich")
    enrich.add_argument("--input", type=Path, required=True)
    enrich.add_argument("--output", type=Path, required=True)
    enrich.add_argument("--audit", type=Path, required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--fit", type=Path, required=True)
    ev.add_argument("--calibration", type=Path, required=True)
    ev.add_argument("--evaluation", type=Path, required=True)
    ev.add_argument("--evaluation-audit", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--scored-output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "enrich":
        return enrich_phase(
            args.input,
            args.output,
            args.audit,
        )
    return evaluate(
        args.fit,
        args.calibration,
        args.evaluation,
        args.evaluation_audit,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
