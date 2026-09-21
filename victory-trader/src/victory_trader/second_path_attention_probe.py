"""Causal one-second path information-value probe for WATCH -> HOT attention."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .attention_flatfile_replay import (
    FLATFILE_CACHE_DIR,
    build_flatfile_scan_day,
)
from .attention_replay import replay_attention
from .attention_runtime import AttentionConfig
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange

SECOND_CACHE_DIR = Path("data/cache/massive-second-attention")
MINUTE_MS = 60_000
SECOND_MS = 1_000
SAMPLE_TICKERS_PER_DAY = 12
ROWS_PER_TICKER = 8
TARGET_HORIZON_MS = 3 * MINUTE_MS
RUNNER_THRESHOLD_PCT = 10.0

BASELINE_FEATURES = [
    "attention_score",
    "attention_rank",
    "return_from_previous_close_pct",
    "minute_body_return_pct",
    "minute_range_pct",
    "log_minute_volume",
    "log_minute_transactions",
    "minute_return_1m_pct",
    "return_accel_1m_pct",
    "volume_ratio_prev1",
    "transactions_ratio_prev1",
]
SECOND_FEATURES = [
    "active_seconds_60",
    "active_seconds_10",
    "last_activity_age_s",
    "sec_last10_return_pct",
    "sec_prev10_return_pct",
    "sec_accel_10_pct",
    "sec_first30_return_pct",
    "sec_last30_return_pct",
    "sec_accel_30_pct",
    "sec_realized_vol_pct",
    "sec_positive_fraction",
    "sec_max_drawdown_pct",
    "sec_max_runup_pct",
    "sec_volume_last10_share",
    "sec_transactions_last10_share",
    "sec_volume_burst_10",
    "sec_transactions_burst_10",
]
MODEL_KWARGS = {
    "learning_rate": 0.05,
    "max_iter": 120,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 17,
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num / den.where(den.gt(0))
    return out.replace([np.inf, -np.inf], np.nan)


def _annotate_scan(scan: pd.DataFrame) -> pd.DataFrame:
    frame = scan.sort_values(["ticker", "t"], kind="stable").copy()
    groups = frame.groupby("ticker", sort=False)
    previous_close = groups["c"].shift()
    previous_return = groups["return_from_previous_close_pct"].shift()
    previous_volume = groups["v"].shift()
    previous_transactions = groups["n"].shift()

    frame["minute_body_return_pct"] = (
        pd.to_numeric(frame["c"], errors="coerce")
        / pd.to_numeric(frame["o"], errors="coerce")
        - 1.0
    ) * 100.0
    frame["minute_range_pct"] = (
        (
            pd.to_numeric(frame["h"], errors="coerce")
            - pd.to_numeric(frame["l"], errors="coerce")
        )
        / pd.to_numeric(frame["c"], errors="coerce")
    ) * 100.0
    frame["log_minute_volume"] = np.log1p(
        pd.to_numeric(frame["v"], errors="coerce").clip(lower=0)
    )
    frame["log_minute_transactions"] = np.log1p(
        pd.to_numeric(frame["n"], errors="coerce").clip(lower=0)
    )
    frame["minute_return_1m_pct"] = (
        pd.to_numeric(frame["c"], errors="coerce") / previous_close - 1.0
    ) * 100.0
    frame["return_accel_1m_pct"] = (
        pd.to_numeric(frame["return_from_previous_close_pct"], errors="coerce")
        - previous_return
    )
    frame["volume_ratio_prev1"] = _safe_ratio(frame["v"], previous_volume)
    frame["transactions_ratio_prev1"] = _safe_ratio(
        frame["n"], previous_transactions
    )

    crossed = (
        pd.to_numeric(
            frame["return_from_previous_close_pct"], errors="coerce"
        ).ge(RUNNER_THRESHOLD_PCT)
    )
    frame["already_runner"] = crossed.groupby(
        frame["ticker"], sort=False
    ).cummax()
    return frame


def _selection_key(day: str, ticker: str) -> str:
    return hashlib.sha256(f"{day}|{ticker}".encode("utf-8")).hexdigest()


def select_watch_rows(
    scan: pd.DataFrame,
    trace: pd.DataFrame,
    *,
    sample_tickers: int = SAMPLE_TICKERS_PER_DAY,
    rows_per_ticker: int = ROWS_PER_TICKER,
) -> pd.DataFrame:
    annotated = _annotate_scan(scan)
    watch = trace.loc[trace["state"].astype(str).eq("watch")].copy()
    merged = watch.merge(
        annotated,
        on=["trading_day", "ticker", "t"],
        how="inner",
        suffixes=("", "_scan"),
        validate="one_to_one",
    )
    merged = merged.loc[~merged["already_runner"].fillna(True)].copy()
    if merged.empty:
        return merged

    day = str(merged["trading_day"].iloc[0])
    tickers = sorted(
        merged["ticker"].astype(str).unique().tolist(),
        key=lambda ticker: _selection_key(day, ticker),
    )[:sample_tickers]
    merged = merged.loc[merged["ticker"].isin(tickers)].copy()

    pieces: list[pd.DataFrame] = []
    for ticker in tickers:
        group = merged.loc[merged["ticker"].eq(ticker)].sort_values("t")
        pieces.append(group.head(rows_per_ticker))
    if not pieces:
        return merged.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _second_frame(payload: dict) -> pd.DataFrame:
    rows = list(payload.get("results") or [])
    if not rows:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "n"])
    frame = pd.DataFrame(rows)
    required = {"t", "o", "h", "l", "c", "v"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"second aggregate response missing columns: {sorted(missing)}")
    if "n" not in frame.columns:
        frame["n"] = np.nan
    for column in ["t", "o", "h", "l", "c", "v", "n"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["t", "c"]).copy()
    frame["t"] = frame["t"].astype("int64")
    frame = frame.sort_values("t", kind="stable").drop_duplicates("t", keep="last")
    return frame.reset_index(drop=True)


def _window_return(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    first_open = float(frame.iloc[0]["o"])
    last_close = float(frame.iloc[-1]["c"])
    if first_open <= 0:
        return np.nan
    return (last_close / first_open - 1.0) * 100.0


def second_path_features(seconds: pd.DataFrame, decision_t: int) -> dict[str, float]:
    if seconds.empty:
        window = seconds
    else:
        window = seconds.loc[
            seconds["t"].ge(decision_t - MINUTE_MS)
            & (seconds["t"] + SECOND_MS).le(decision_t)
        ].copy()

    last10 = window.loc[window["t"].ge(decision_t - 10_000)]
    prev10 = window.loc[
        window["t"].ge(decision_t - 20_000)
        & window["t"].lt(decision_t - 10_000)
    ]
    first30 = window.loc[window["t"].lt(decision_t - 30_000)]
    last30 = window.loc[window["t"].ge(decision_t - 30_000)]

    close = pd.to_numeric(window.get("c", pd.Series(dtype=float)), errors="coerce")
    log_returns = np.log(close.where(close.gt(0))).diff().dropna()
    positive_fraction = (
        float(log_returns.gt(0).mean()) if not log_returns.empty else np.nan
    )
    realized_vol = (
        float(log_returns.std(ddof=0) * 100.0)
        if not log_returns.empty
        else np.nan
    )
    if not close.dropna().empty:
        valid_close = close.dropna()
        running_high = valid_close.cummax()
        drawdown = (valid_close / running_high - 1.0) * 100.0
        first_open = float(window.iloc[0]["o"])
        runup = (
            (valid_close / first_open - 1.0) * 100.0
            if first_open > 0
            else pd.Series(dtype=float)
        )
        max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan
        max_runup = float(runup.max()) if len(runup) else np.nan
        last_activity_age = float(
            max(decision_t - SECOND_MS - int(window.iloc[-1]["t"]), 0)
            / 1_000.0
        )
    else:
        max_drawdown = np.nan
        max_runup = np.nan
        last_activity_age = np.nan

    volume_total = float(pd.to_numeric(window.get("v"), errors="coerce").sum())
    volume_last10 = float(pd.to_numeric(last10.get("v"), errors="coerce").sum())
    volume_prev10 = float(pd.to_numeric(prev10.get("v"), errors="coerce").sum())
    tx_total = float(pd.to_numeric(window.get("n"), errors="coerce").sum())
    tx_last10 = float(pd.to_numeric(last10.get("n"), errors="coerce").sum())
    tx_prev10 = float(pd.to_numeric(prev10.get("n"), errors="coerce").sum())

    last10_return = _window_return(last10)
    prev10_return = _window_return(prev10)
    first30_return = _window_return(first30)
    last30_return = _window_return(last30)

    return {
        "active_seconds_60": float(len(window)),
        "active_seconds_10": float(len(last10)),
        "last_activity_age_s": last_activity_age,
        "sec_last10_return_pct": last10_return,
        "sec_prev10_return_pct": prev10_return,
        "sec_accel_10_pct": (
            last10_return - prev10_return
            if np.isfinite(last10_return) and np.isfinite(prev10_return)
            else np.nan
        ),
        "sec_first30_return_pct": first30_return,
        "sec_last30_return_pct": last30_return,
        "sec_accel_30_pct": (
            last30_return - first30_return
            if np.isfinite(last30_return) and np.isfinite(first30_return)
            else np.nan
        ),
        "sec_realized_vol_pct": realized_vol,
        "sec_positive_fraction": positive_fraction,
        "sec_max_drawdown_pct": max_drawdown,
        "sec_max_runup_pct": max_runup,
        "sec_volume_last10_share": (
            volume_last10 / volume_total if volume_total > 0 else np.nan
        ),
        "sec_transactions_last10_share": (
            tx_last10 / tx_total if tx_total > 0 else np.nan
        ),
        "sec_volume_burst_10": (
            volume_last10 / volume_prev10 if volume_prev10 > 0 else np.nan
        ),
        "sec_transactions_burst_10": (
            tx_last10 / tx_prev10 if tx_prev10 > 0 else np.nan
        ),
    }


def _target_for_row(scan_ticker: pd.DataFrame, decision_t: int, current_close: float) -> tuple[float, int]:
    future = scan_ticker.loc[
        scan_ticker["t"].gt(decision_t)
        & scan_ticker["t"].le(decision_t + TARGET_HORIZON_MS)
    ]
    if future.empty or current_close <= 0:
        return np.nan, 0
    returns = (
        pd.to_numeric(future["c"], errors="coerce") / current_close - 1.0
    ) * 100.0
    returns = returns.dropna()
    if returns.empty:
        return np.nan, 0
    return float(returns.max()), int(len(returns))


def build_day_examples(
    scan: pd.DataFrame,
    trace: pd.DataFrame,
    client: MassiveClient,
    day: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = select_watch_rows(scan, trace)
    if selected.empty:
        return selected, {
            "trading_day": day.isoformat(),
            "selected_tickers": 0,
            "candidate_rows": 0,
            "eligible_rows": 0,
        }

    second_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker in sorted(selected["ticker"].astype(str).unique()):
        payload = client.second_bars_range(ticker, day, day, adjusted=False)
        second_by_ticker[ticker] = _second_frame(payload)

    scan_by_ticker = {
        ticker: group.sort_values("t")
        for ticker, group in scan.groupby("ticker", sort=False)
    }
    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        ticker = str(row.ticker)
        decision_t = int(row.t)
        current_close = float(row.c)
        target, future_points = _target_for_row(
            scan_by_ticker[ticker], decision_t, current_close
        )
        if not np.isfinite(target):
            continue
        item = row._asdict()
        item.update(second_path_features(second_by_ticker[ticker], decision_t))
        item["future_max_3m_return_pct"] = target
        item["future_points_3m"] = future_points
        rows.append(item)

    examples = pd.DataFrame(rows)
    return examples, {
        "trading_day": day.isoformat(),
        "selected_tickers": int(selected["ticker"].nunique()),
        "candidate_rows": int(len(selected)),
        "eligible_rows": int(len(examples)),
        "second_rows": int(sum(len(frame) for frame in second_by_ticker.values())),
        "second_nonempty_tickers": int(
            sum(not frame.empty for frame in second_by_ticker.values())
        ),
    }


def _spearman(y: pd.Series, pred: np.ndarray | pd.Series) -> float | None:
    left = pd.Series(y).reset_index(drop=True)
    right = pd.Series(pred).reset_index(drop=True)
    value = left.corr(right, method="spearman")
    return float(value) if pd.notna(value) else None


def _metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float | int | None]:
    y = pd.to_numeric(frame["future_max_3m_return_pct"], errors="coerce")
    pred = pd.to_numeric(frame[prediction_column], errors="coerce")
    valid = y.notna() & pred.notna()
    y = y.loc[valid]
    pred = pred.loc[valid]
    if y.empty:
        return {"rows": 0, "spearman": None, "mae": None, "top_quartile_target_mean": None}
    cutoff = float(pred.quantile(0.75))
    top = y.loc[pred.ge(cutoff)]
    return {
        "rows": int(len(y)),
        "spearman": _spearman(y, pred),
        "mae": float(np.mean(np.abs(y.to_numpy() - pred.to_numpy()))),
        "top_quartile_target_mean": float(top.mean()) if len(top) else None,
    }


def _target_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    y = pd.to_numeric(frame["future_max_3m_return_pct"], errors="coerce").dropna()
    if y.empty:
        return {"rows": 0}
    return {
        "rows": int(len(y)),
        "tickers": int(frame.loc[y.index, "ticker"].nunique()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "p75": float(y.quantile(0.75)),
        "p90": float(y.quantile(0.90)),
    }


def fit_and_evaluate(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    days = sorted(dataset["trading_day"].astype(str).unique())
    if len(days) < 6:
        raise ValueError(f"need at least six trading sessions, found {len(days)}")
    fit_days = days[:-2]
    eval_days = days[-2:]
    work = dataset.copy()
    work["split"] = np.where(
        work["trading_day"].astype(str).isin(eval_days), "eval", "fit"
    )
    fit = work.loc[work["split"].eq("fit")].copy()
    evaluation = work.loc[work["split"].eq("eval")].copy()
    if len(fit) < 40 or len(evaluation) < 40:
        raise ValueError(
            f"insufficient rows for frozen fit/eval: fit={len(fit)} eval={len(evaluation)}"
        )

    baseline = HistGradientBoostingRegressor(**MODEL_KWARGS)
    extended = HistGradientBoostingRegressor(**MODEL_KWARGS)
    baseline.fit(
        fit[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["future_max_3m_return_pct"],
    )
    extended.fit(
        fit[BASELINE_FEATURES + SECOND_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["future_max_3m_return_pct"],
    )
    work["baseline_prediction"] = baseline.predict(
        work[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )
    work["extended_prediction"] = extended.predict(
        work[BASELINE_FEATURES + SECOND_FEATURES].replace(
            [np.inf, -np.inf], np.nan
        )
    )
    evaluation = work.loc[work["split"].eq("eval")].copy()

    pooled_baseline = _metrics(evaluation, "baseline_prediction")
    pooled_extended = _metrics(evaluation, "extended_prediction")
    by_day: dict[str, object] = {}
    nonlower_each_day = True
    min_rows_each_day = True
    for day in eval_days:
        day_frame = evaluation.loc[evaluation["trading_day"].astype(str).eq(day)]
        baseline_metrics = _metrics(day_frame, "baseline_prediction")
        extended_metrics = _metrics(day_frame, "extended_prediction")
        b = baseline_metrics["spearman"]
        e = extended_metrics["spearman"]
        if b is None or e is None or e < b:
            nonlower_each_day = False
        if len(day_frame) < 30:
            min_rows_each_day = False
        by_day[day] = {
            "target": _target_summary(day_frame),
            "baseline": baseline_metrics,
            "extended": extended_metrics,
            "incremental_spearman": (
                float(e - b) if b is not None and e is not None else None
            ),
        }

    pooled_b = pooled_baseline["spearman"]
    pooled_e = pooled_extended["spearman"]
    top_b = pooled_baseline["top_quartile_target_mean"]
    top_e = pooled_extended["top_quartile_target_mean"]
    gate = bool(
        min_rows_each_day
        and pooled_b is not None
        and pooled_e is not None
        and pooled_e > pooled_b
        and nonlower_each_day
        and top_b is not None
        and top_e is not None
        and top_e >= top_b
    )
    return work, {
        "fit_days": fit_days,
        "eval_days": eval_days,
        "fit_rows": int(len(fit)),
        "eval_rows": int(len(evaluation)),
        "target": _target_summary(evaluation),
        "baseline": pooled_baseline,
        "extended": pooled_extended,
        "incremental_spearman": (
            float(pooled_e - pooled_b)
            if pooled_b is not None and pooled_e is not None
            else None
        ),
        "by_day": by_day,
        "information_value_gate_pass": gate,
    }


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    day_summaries: list[dict[str, object]] = []
    pieces: list[pd.DataFrame] = []
    for day in daterange(start, end):
        if not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        trace = replay_attention(scan, AttentionConfig())
        examples, sample_summary = build_day_examples(
            scan, trace, second_client, day
        )
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
        raise ValueError("second-path probe produced no eligible examples")
    dataset = pd.concat(pieces, ignore_index=True)
    modeled, evaluation = fit_and_evaluate(dataset)
    second_counts = pd.to_numeric(
        modeled["active_seconds_60"], errors="coerce"
    ).fillna(0)
    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sample_tickers_per_day": SAMPLE_TICKERS_PER_DAY,
        "rows_per_ticker": ROWS_PER_TICKER,
        "target_horizon_minutes": 3,
        "baseline_features": BASELINE_FEATURES,
        "second_features": SECOND_FEATURES,
        "model": MODEL_KWARGS,
        "days": day_summaries,
        "dataset_rows": int(len(modeled)),
        "dataset_tickers": int(modeled["ticker"].nunique()),
        "rows_with_any_second_data": int(second_counts.gt(0).sum()),
        "second_data_row_coverage": float(second_counts.gt(0).mean()),
        "median_active_seconds_60": float(second_counts.median()),
        "second_client_stats": second_client.stats.to_dict(),
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    return modeled, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe incremental one-second path information for WATCH -> HOT."
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
    scan_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    second_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    dataset, summary = run_probe(
        store,
        scan_client,
        second_client,
        args.start,
        args.end,
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
