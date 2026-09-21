"""Market-relative completed-minute information probe for WATCH episode value."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import replay_attention
from .attention_runtime import AttentionConfig
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange
from .second_path_attention_probe import (
    BASELINE_FEATURES,
    MODEL_KWARGS,
    _annotate_scan,
)
from .watch_episode_attention_probe import (
    _episode_metrics,
    _episode_target,
    _episode_target_summary,
    enumerate_watch_episodes,
)

FIT_END = "2026-01-16"
EVAL_START = "2026-01-20"
EVAL_END = "2026-01-26"

MARKET_RELATIVE_FEATURES = [
    "trailing_return_3m_pct",
    "trailing_return_5m_pct",
    "return_accel_vs_prev1_pct",
    "volume_accel_1m_vs_prior20m",
    "volume_accel_5m_vs_prior20m",
    "transactions_accel_1m_vs_prior20m",
    "transactions_accel_5m_vs_prior20m",
    "range_expansion_vs_prior20m",
    "market_rank_return_1m",
    "market_rank_return_3m",
    "market_rank_return_5m",
    "market_rank_return_accel_1m",
    "market_rank_volume_accel_1m",
    "market_rank_volume_accel_5m",
    "market_rank_transactions_accel_1m",
    "market_rank_transactions_accel_5m",
    "market_rank_range_expansion",
    "market_positive_return_1m_frac",
    "market_positive_return_5m_frac",
    "market_median_return_1m_pct",
    "market_median_return_5m_pct",
    "market_median_volume_accel_1m",
    "market_median_transactions_accel_1m",
]


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return (num / den.where(den.gt(0))).replace([np.inf, -np.inf], np.nan)


def _group_rolling_mean(
    frame: pd.DataFrame,
    column: str,
    *,
    window: int,
    shift: int,
    min_periods: int,
) -> pd.Series:
    return (
        frame.groupby("ticker", sort=False)[column]
        .transform(
            lambda series: (
                pd.to_numeric(series, errors="coerce")
                .shift(shift)
                .rolling(window, min_periods=min_periods)
                .mean()
            )
        )
    )


def _group_rolling_median(
    frame: pd.DataFrame,
    column: str,
    *,
    window: int,
    shift: int,
    min_periods: int,
) -> pd.Series:
    return (
        frame.groupby("ticker", sort=False)[column]
        .transform(
            lambda series: (
                pd.to_numeric(series, errors="coerce")
                .shift(shift)
                .rolling(window, min_periods=min_periods)
                .median()
            )
        )
    )


def enrich_market_relative_features(scan: pd.DataFrame) -> pd.DataFrame:
    """Add strictly causal ticker-history and same-timestamp market context."""

    frame = _annotate_scan(scan)
    frame = frame.sort_values(["ticker", "t"], kind="stable").copy()
    group = frame.groupby("ticker", sort=False)

    close = pd.to_numeric(frame["c"], errors="coerce")
    lag3 = group["c"].shift(3)
    lag5 = group["c"].shift(5)
    frame["trailing_return_3m_pct"] = (close / lag3 - 1.0) * 100.0
    frame["trailing_return_5m_pct"] = (close / lag5 - 1.0) * 100.0

    previous_1m_return = group["minute_return_1m_pct"].shift(1)
    frame["return_accel_vs_prev1_pct"] = (
        pd.to_numeric(frame["minute_return_1m_pct"], errors="coerce")
        - pd.to_numeric(previous_1m_return, errors="coerce")
    )

    volume = pd.to_numeric(frame["v"], errors="coerce")
    transactions = pd.to_numeric(frame["n"], errors="coerce")
    prior20_volume = _group_rolling_mean(
        frame, "v", window=20, shift=1, min_periods=5
    )
    prior20_transactions = _group_rolling_mean(
        frame, "n", window=20, shift=1, min_periods=5
    )
    recent5_volume = _group_rolling_mean(
        frame, "v", window=5, shift=0, min_periods=2
    )
    recent5_transactions = _group_rolling_mean(
        frame, "n", window=5, shift=0, min_periods=2
    )
    preceding20_volume = _group_rolling_mean(
        frame, "v", window=20, shift=5, min_periods=5
    )
    preceding20_transactions = _group_rolling_mean(
        frame, "n", window=20, shift=5, min_periods=5
    )
    frame["volume_accel_1m_vs_prior20m"] = _safe_ratio(
        volume, prior20_volume
    )
    frame["volume_accel_5m_vs_prior20m"] = _safe_ratio(
        recent5_volume, preceding20_volume
    )
    frame["transactions_accel_1m_vs_prior20m"] = _safe_ratio(
        transactions, prior20_transactions
    )
    frame["transactions_accel_5m_vs_prior20m"] = _safe_ratio(
        recent5_transactions, preceding20_transactions
    )

    prior20_range = _group_rolling_median(
        frame, "minute_range_pct", window=20, shift=1, min_periods=5
    )
    frame["range_expansion_vs_prior20m"] = _safe_ratio(
        frame["minute_range_pct"], prior20_range
    )

    rank_sources = {
        "market_rank_return_1m": "minute_return_1m_pct",
        "market_rank_return_3m": "trailing_return_3m_pct",
        "market_rank_return_5m": "trailing_return_5m_pct",
        "market_rank_return_accel_1m": "return_accel_vs_prev1_pct",
        "market_rank_volume_accel_1m": "volume_accel_1m_vs_prior20m",
        "market_rank_volume_accel_5m": "volume_accel_5m_vs_prior20m",
        "market_rank_transactions_accel_1m": (
            "transactions_accel_1m_vs_prior20m"
        ),
        "market_rank_transactions_accel_5m": (
            "transactions_accel_5m_vs_prior20m"
        ),
        "market_rank_range_expansion": "range_expansion_vs_prior20m",
    }
    cross = frame.groupby(["trading_day", "t"], sort=False)
    for output, source in rank_sources.items():
        frame[output] = cross[source].rank(method="average", pct=True)

    return1 = pd.to_numeric(frame["minute_return_1m_pct"], errors="coerce")
    return5 = pd.to_numeric(frame["trailing_return_5m_pct"], errors="coerce")
    frame["_positive_return_1m"] = return1.gt(0).where(return1.notna()).astype(float)
    frame["_positive_return_5m"] = return5.gt(0).where(return5.notna()).astype(float)
    cross = frame.groupby(["trading_day", "t"], sort=False)
    frame["market_positive_return_1m_frac"] = cross[
        "_positive_return_1m"
    ].transform("mean")
    frame["market_positive_return_5m_frac"] = cross[
        "_positive_return_5m"
    ].transform("mean")
    frame["market_median_return_1m_pct"] = cross[
        "minute_return_1m_pct"
    ].transform("median")
    frame["market_median_return_5m_pct"] = cross[
        "trailing_return_5m_pct"
    ].transform("median")
    frame["market_median_volume_accel_1m"] = cross[
        "volume_accel_1m_vs_prior20m"
    ].transform("median")
    frame["market_median_transactions_accel_1m"] = cross[
        "transactions_accel_1m_vs_prior20m"
    ].transform("median")
    frame = frame.drop(columns=["_positive_return_1m", "_positive_return_5m"])

    for column in MARKET_RELATIVE_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_day_examples(
    scan: pd.DataFrame,
    trace: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    enriched = enrich_market_relative_features(scan)
    scan_index = enriched.set_index(["ticker", "t"], drop=False)
    trace_index = trace.set_index(["ticker", "t"], drop=False)
    scan_by_ticker = {
        ticker: group.sort_values("t", kind="stable")
        for ticker, group in enriched.groupby("ticker", sort=False)
    }

    rows: list[dict[str, object]] = []
    episode_count = 0
    for episode in enumerate_watch_episodes(trace):
        key = (episode.ticker, episode.entry_t)
        if key not in scan_index.index or key not in trace_index.index:
            continue
        scan_row = scan_index.loc[key]
        trace_row = trace_index.loc[key]
        if isinstance(scan_row, pd.DataFrame) or isinstance(trace_row, pd.DataFrame):
            raise ValueError("WATCH-entry key is not unique")
        if bool(scan_row["already_runner"]):
            continue

        episode_count += 1
        peak, runner, future_points = _episode_target(
            scan_by_ticker[episode.ticker],
            entry_t=int(episode.entry_t),
            end_t=int(episode.end_t),
            end_exclusive=bool(episode.end_exclusive),
            entry_close=float(scan_row["c"]),
        )
        item = scan_row.to_dict()
        item["attention_score"] = float(trace_row["attention_score"])
        item["attention_rank"] = trace_row["attention_rank"]
        item["episode_end_t"] = int(episode.end_t)
        item["episode_end_exclusive"] = bool(episode.end_exclusive)
        item["episode_ordinal"] = int(episode.ordinal)
        item["remaining_episode_peak_return_pct"] = peak
        item["runner_in_remaining_episode"] = runner
        item["episode_future_points"] = future_points
        item["episode_duration_minutes"] = max(
            (int(episode.end_t) - int(episode.entry_t)) / 60_000.0,
            0.0,
        )
        rows.append(item)

    examples = pd.DataFrame(rows)
    return examples, {
        "eligible_episodes": int(episode_count),
        "eligible_tickers": (
            int(examples["ticker"].nunique()) if not examples.empty else 0
        ),
    }


def _metrics_with_runner(
    frame: pd.DataFrame,
    prediction_column: str,
) -> dict[str, object]:
    result = _episode_metrics(frame, prediction_column)
    pred = pd.to_numeric(frame[prediction_column], errors="coerce")
    valid = pred.notna()
    if not valid.any():
        result["top_quartile_runner_rate"] = None
        return result
    cutoff = float(pred.loc[valid].quantile(0.75))
    top = frame.loc[valid & pred.ge(cutoff)]
    result["top_quartile_runner_rate"] = (
        float(
            top["runner_in_remaining_episode"]
            .fillna(False)
            .astype(bool)
            .mean()
        )
        if len(top)
        else None
    )
    return result


def fit_and_evaluate(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    work = dataset.copy()
    day = work["trading_day"].astype(str)
    is_fit = day.le(FIT_END)
    is_eval = day.ge(EVAL_START) & day.le(EVAL_END)
    if (~(is_fit | is_eval)).any():
        raise ValueError("dataset contains dates outside frozen fit/eval windows")
    work["split"] = np.where(is_eval, "eval", "fit")
    fit = work.loc[work["split"].eq("fit")].copy()
    evaluation = work.loc[work["split"].eq("eval")].copy()
    eval_days = sorted(evaluation["trading_day"].astype(str).unique())
    if len(eval_days) != 5:
        raise ValueError(f"expected five evaluation sessions, found {eval_days}")

    target = "remaining_episode_peak_return_pct"
    baseline = HistGradientBoostingRegressor(**MODEL_KWARGS)
    extended = HistGradientBoostingRegressor(**MODEL_KWARGS)
    baseline.fit(
        fit[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit[target],
    )
    extended.fit(
        fit[BASELINE_FEATURES + MARKET_RELATIVE_FEATURES].replace(
            [np.inf, -np.inf], np.nan
        ),
        fit[target],
    )
    work["baseline_prediction"] = baseline.predict(
        work[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )
    work["extended_prediction"] = extended.predict(
        work[BASELINE_FEATURES + MARKET_RELATIVE_FEATURES].replace(
            [np.inf, -np.inf], np.nan
        )
    )
    evaluation = work.loc[work["split"].eq("eval")].copy()

    pooled_baseline = _metrics_with_runner(
        evaluation, "baseline_prediction"
    )
    pooled_extended = _metrics_with_runner(
        evaluation, "extended_prediction"
    )

    by_day: dict[str, object] = {}
    increments: list[float] = []
    nonlower_days = 0
    minimum_rows = True
    for eval_day in eval_days:
        frame = evaluation.loc[
            evaluation["trading_day"].astype(str).eq(eval_day)
        ]
        b = _metrics_with_runner(frame, "baseline_prediction")
        e = _metrics_with_runner(frame, "extended_prediction")
        b_s = b.get("spearman")
        e_s = e.get("spearman")
        diff = None
        if b_s is not None and e_s is not None:
            diff = float(e_s - b_s)
            increments.append(diff)
            if e_s >= b_s:
                nonlower_days += 1
        if len(frame) < 100:
            minimum_rows = False
        by_day[eval_day] = {
            "target": _episode_target_summary(frame),
            "baseline": b,
            "extended": e,
            "incremental_spearman": diff,
        }

    pooled_b = pooled_baseline.get("spearman")
    pooled_e = pooled_extended.get("spearman")
    mae_b = pooled_baseline.get("mae")
    mae_e = pooled_extended.get("mae")
    top_b = pooled_baseline.get("top_quartile_target_mean")
    top_e = pooled_extended.get("top_quartile_target_mean")
    run_b = pooled_baseline.get("top_quartile_runner_rate")
    run_e = pooled_extended.get("top_quartile_runner_rate")
    mean_increment = (
        float(np.mean(increments)) if len(increments) == len(eval_days) else None
    )
    gate = bool(
        minimum_rows
        and pooled_b is not None
        and pooled_e is not None
        and pooled_e > pooled_b
        and mean_increment is not None
        and mean_increment > 0
        and nonlower_days >= 4
        and mae_b is not None
        and mae_e is not None
        and mae_e <= mae_b
        and top_b is not None
        and top_e is not None
        and top_e >= top_b
        and run_b is not None
        and run_e is not None
        and run_e >= run_b
    )

    missingness = {
        feature: float(
            pd.to_numeric(work[feature], errors="coerce").isna().mean()
        )
        for feature in MARKET_RELATIVE_FEATURES
    }
    return work, {
        "fit_days": sorted(fit["trading_day"].astype(str).unique()),
        "eval_days": eval_days,
        "fit_rows": int(len(fit)),
        "eval_rows": int(len(evaluation)),
        "target": _episode_target_summary(evaluation),
        "baseline": pooled_baseline,
        "extended": pooled_extended,
        "incremental_spearman": (
            float(pooled_e - pooled_b)
            if pooled_b is not None and pooled_e is not None
            else None
        ),
        "mean_day_incremental_spearman": mean_increment,
        "nonlower_eval_days": nonlower_days,
        "by_day": by_day,
        "market_relative_feature_missingness": missingness,
        "information_value_gate_pass": gate,
    }


def run_probe(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pieces: list[pd.DataFrame] = []
    day_summaries: list[dict[str, object]] = []
    for day in daterange(start, end):
        if not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, rest_client, day)
        trace = replay_attention(scan, AttentionConfig())
        examples, sample_summary = build_day_examples(scan, trace)
        sample_summary["trading_day"] = day.isoformat()
        sample_summary["scan_rows"] = int(len(scan))
        sample_summary["trace_rows"] = int(len(trace))
        sample_summary["runner_crossings"] = int(
            scan["runner_cross_now"].fillna(False).astype(bool).sum()
        )
        sample_summary["point_in_time_common_stocks"] = scan_summary.get(
            "point_in_time_common_stocks"
        )
        day_summaries.append(sample_summary)
        if not examples.empty:
            pieces.append(examples)

    if not pieces:
        raise ValueError("market-relative probe produced no examples")
    dataset = pd.concat(pieces, ignore_index=True)
    modeled, evaluation = fit_and_evaluate(dataset)
    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fit_end": FIT_END,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "population": "all eligible pre-runner WATCH-entry episodes",
        "baseline_features": BASELINE_FEATURES,
        "market_relative_features": MARKET_RELATIVE_FEATURES,
        "model": MODEL_KWARGS,
        "days": day_summaries,
        "dataset_rows": int(len(modeled)),
        "dataset_tickers": int(modeled["ticker"].nunique()),
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    return modeled, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate market-relative acceleration for WATCH episode value."
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR,
    )
    rest_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    dataset, summary = run_probe(
        store, rest_client, args.start, args.end
    )
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.dataset_output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
