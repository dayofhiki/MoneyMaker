"""Request 173: market-regime context for post-entry remaining value.

No new dates and no new external data. Reuse Request-171 rich-second POSITION
states plus the already-opened Request-163 market-wide minute scan. At each
causal POSITION state attach only same-completed-minute cross-sectional market
context.

The intervention tests whether a continuously scanning trader benefits from
knowing whether momentum is broad/healthy or isolated/exhausted while managing
an open position.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .buy_gated_transition_option_value import evaluate_option_signal
from .rich_second_position_value import (
    RICH_SECOND_FEATURES,
    predict_rich_option,
    train_rich_option,
)
from .second_path_attention_probe import _annotate_scan
from .selected_hot_position_value_observability import (
    MODEL_FEATURES,
    OPTION_SEED,
    apply_excess_target,
    fit_minute_baselines,
)

REQUEST_ID = 173

MARKET_REGIME_FEATURES = (
    "market_universe_count",
    "market_runner_5_fraction",
    "market_runner_10_fraction",
    "market_runner_20_fraction",
    "market_return_median_pct",
    "market_return_p90_pct",
    "market_return_p99_pct",
    "market_return_dispersion_pct",
    "market_body_positive_fraction",
    "market_1m_positive_fraction",
    "market_attention_top1",
    "market_attention_top10_mean",
    "market_attention_p90",
    "market_log_total_volume",
    "market_log_total_transactions",
    "market_volume_top10_share",
)

MIN_MARKET_CONTEXT_COVERAGE = 0.95
MIN_POOLED_SPEARMAN_IMPROVEMENT = 0.02
MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT = 0.01
MIN_SELECTED_RATE = 0.10
MIN_GOOD_DAYS = 4


@dataclass(frozen=True)
class MarketRegimeOptionModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def build_market_regime(scan: pd.DataFrame) -> pd.DataFrame:
    annotated = _annotate_scan(scan)
    rows: list[dict[str, float | int | str]] = []
    for (day, timestamp), group in annotated.groupby(
        ["trading_day", "t"],
        sort=True,
    ):
        ret = pd.to_numeric(
            group["return_from_previous_close_pct"],
            errors="coerce",
        ).dropna()
        body = pd.to_numeric(
            group["minute_body_return_pct"],
            errors="coerce",
        ).dropna()
        one = pd.to_numeric(
            group["minute_return_1m_pct"],
            errors="coerce",
        ).dropna()
        attention = pd.to_numeric(
            group["attention_score"],
            errors="coerce",
        ).dropna()
        volume = pd.to_numeric(
            group["v"], errors="coerce"
        ).fillna(0.0).clip(lower=0)
        tx = pd.to_numeric(
            group["n"], errors="coerce"
        ).fillna(0.0).clip(lower=0)

        n = max(len(group), 1)
        total_volume = float(volume.sum())
        sorted_volume = np.sort(volume.to_numpy(dtype=float))[::-1]
        top10_volume = float(sorted_volume[:10].sum())

        sorted_attention = np.sort(
            attention.to_numpy(dtype=float)
        )[::-1]
        top1_attention = (
            float(sorted_attention[0])
            if len(sorted_attention)
            else np.nan
        )
        top10_attention = (
            float(np.mean(sorted_attention[:10]))
            if len(sorted_attention)
            else np.nan
        )

        rows.append(
            {
                "trading_day": str(day),
                "state_t": int(timestamp),
                "market_universe_count": float(len(group)),
                "market_runner_5_fraction": (
                    float(ret.ge(5.0).sum() / n)
                    if len(ret)
                    else np.nan
                ),
                "market_runner_10_fraction": (
                    float(ret.ge(10.0).sum() / n)
                    if len(ret)
                    else np.nan
                ),
                "market_runner_20_fraction": (
                    float(ret.ge(20.0).sum() / n)
                    if len(ret)
                    else np.nan
                ),
                "market_return_median_pct": (
                    float(ret.median()) if len(ret) else np.nan
                ),
                "market_return_p90_pct": (
                    float(ret.quantile(0.90)) if len(ret) else np.nan
                ),
                "market_return_p99_pct": (
                    float(ret.quantile(0.99)) if len(ret) else np.nan
                ),
                "market_return_dispersion_pct": (
                    float(ret.std(ddof=0)) if len(ret) else np.nan
                ),
                "market_body_positive_fraction": (
                    float(body.gt(0).mean()) if len(body) else np.nan
                ),
                "market_1m_positive_fraction": (
                    float(one.gt(0).mean()) if len(one) else np.nan
                ),
                "market_attention_top1": top1_attention,
                "market_attention_top10_mean": top10_attention,
                "market_attention_p90": (
                    float(attention.quantile(0.90))
                    if len(attention)
                    else np.nan
                ),
                "market_log_total_volume": float(
                    np.log1p(total_volume)
                ),
                "market_log_total_transactions": float(
                    np.log1p(float(tx.sum()))
                ),
                "market_volume_top10_share": (
                    top10_volume / total_volume
                    if total_volume > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def attach_market_regime(
    frame: pd.DataFrame,
    regime: pd.DataFrame,
) -> pd.DataFrame:
    left = frame.copy()
    left["trading_day"] = left["trading_day"].astype(str)
    left["state_t"] = pd.to_numeric(
        left["state_t"], errors="raise"
    ).astype("int64")
    return left.merge(
        regime,
        on=["trading_day", "state_t"],
        how="left",
        validate="many_to_one",
    )


def _columns() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *MODEL_FEATURES,
                *RICH_SECOND_FEATURES,
                *MARKET_REGIME_FEATURES,
            ]
        )
    )


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reindex(columns=_columns()).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_market_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> MarketRegimeOptionModel:
    target = pd.to_numeric(
        fit["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError("request 173 insufficient fit support")

    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=OPTION_SEED + 20,
    )
    model.fit(
        _feature_frame(train),
        y.clip(lower=low, upper=high),
    )

    cal_target = pd.to_numeric(
        calibration["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    if int(cal_valid.sum()) < 500:
        raise ValueError("request 173 insufficient calibration support")
    raw = model.predict(_feature_frame(calibration.loc[cal_valid]))
    offset = float(
        cal_target.loc[cal_valid].mean() - float(np.mean(raw))
    )
    return MarketRegimeOptionModel(
        model=model,
        feature_columns=_columns(),
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_market_model(
    frame: pd.DataFrame,
    fitted: MarketRegimeOptionModel,
) -> np.ndarray:
    return fitted.model.predict(_feature_frame(frame)) + fitted.offset


def run(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    scan_path: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    evaluation = pd.read_parquet(evaluation_path)
    scan = pd.read_parquet(scan_path)
    wanted_days = set(
        fit["trading_day"].astype(str).unique().tolist()
        + calibration["trading_day"].astype(str).unique().tolist()
        + evaluation["trading_day"].astype(str).unique().tolist()
    )
    scan = scan.loc[
        scan["trading_day"].astype(str).isin(wanted_days)
    ].copy()

    regime = build_market_regime(scan)
    fit = attach_market_regime(fit, regime)
    calibration = attach_market_regime(calibration, regime)
    evaluation = attach_market_regime(evaluation, regime)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    rich_model = train_rich_option(fit, calibration)
    market_model = train_market_model(fit, calibration)

    scored = evaluation.copy()
    scored["rich_second_option_score"] = predict_rich_option(
        scored, rich_model
    )
    scored["market_regime_option_score"] = predict_market_model(
        scored, market_model
    )

    baseline = evaluate_option_signal(
        scored,
        "rich_second_option_score",
        label="request171_rich_second",
    )
    candidate = evaluate_option_signal(
        scored,
        "market_regime_option_score",
        label="rich_second_plus_market_regime",
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
    context_coverage = float(
        evaluation.loc[:, MARKET_REGIME_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )

    promotion = bool(
        context_coverage >= MIN_MARKET_CONTEXT_COVERAGE
        and pooled_improvement is not None
        and pooled_improvement
        >= MIN_POOLED_SPEARMAN_IMPROVEMENT
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
        "market_regime_features": list(MARKET_REGIME_FEATURES),
        "market_context_coverage": context_coverage,
        "baseline": baseline,
        "candidate": candidate,
        "pooled_spearman_improvement": pooled_improvement,
        "same_minute_median_improvement": minute_improvement,
        "model_diagnostics": {
            "feature_count": len(market_model.feature_columns),
            "offset": market_model.offset,
            "winsor_low_pct": market_model.winsor_low,
            "winsor_high_pct": market_model.winsor_high,
        },
        "frozen_gate": {
            "min_market_context_coverage": MIN_MARKET_CONTEXT_COVERAGE,
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
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.fit,
        args.calibration,
        args.evaluation,
        args.scan,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
