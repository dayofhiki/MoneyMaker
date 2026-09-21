"""Adaptive WATCH-episode attention-value probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
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
    SECOND_CACHE_DIR,
    SECOND_FEATURES,
    _annotate_scan,
    _metrics,
    _second_frame,
    _target_summary,
    second_path_features,
)

SAMPLE_TICKERS_PER_DAY = 24
MAX_EPISODES_PER_TICKER = 2
FIT_END = "2026-01-09"
EVAL_START = "2026-01-12"
EVAL_END = "2026-01-16"
FOCUS_STATES = {"watch", "hot"}


@dataclass(frozen=True)
class WatchEpisode:
    ticker: str
    entry_t: int
    end_t: int
    end_exclusive: bool
    ordinal: int


def _selection_key(day: str, ticker: str) -> str:
    return hashlib.sha256(f"{day}|{ticker}".encode("utf-8")).hexdigest()


def enumerate_watch_episodes(trace: pd.DataFrame) -> list[WatchEpisode]:
    """Enumerate focus episodes that begin with a WATCH transition."""

    episodes: list[WatchEpisode] = []
    for ticker, group in trace.groupby("ticker", sort=False):
        ordered = group.sort_values("t", kind="stable")
        in_focus = False
        candidate_start: int | None = None
        candidate_ordinal = 0
        last_focus_t: int | None = None

        for row in ordered.itertuples(index=False):
            state = str(row.state)
            timestamp = int(row.t)
            focus = state in FOCUS_STATES

            if not in_focus and focus:
                in_focus = True
                last_focus_t = timestamp
                if state == "watch":
                    candidate_ordinal += 1
                    candidate_start = timestamp
                else:
                    # Direct-HOT episodes are already beyond the transition
                    # this diagnostic is designed to learn.
                    candidate_start = None
                continue

            if in_focus and focus:
                last_focus_t = timestamp
                continue

            if in_focus and not focus:
                if candidate_start is not None and last_focus_t is not None:
                    episodes.append(
                        WatchEpisode(
                            ticker=str(ticker),
                            entry_t=candidate_start,
                            end_t=timestamp,
                            end_exclusive=True,
                            ordinal=candidate_ordinal,
                        )
                    )
                in_focus = False
                candidate_start = None
                last_focus_t = None

        if in_focus and candidate_start is not None and last_focus_t is not None:
            episodes.append(
                WatchEpisode(
                    ticker=str(ticker),
                    entry_t=candidate_start,
                    end_t=last_focus_t,
                    end_exclusive=False,
                    ordinal=candidate_ordinal,
                )
            )
    return episodes


def _episode_target(
    scan_ticker: pd.DataFrame,
    *,
    entry_t: int,
    end_t: int,
    end_exclusive: bool,
    entry_close: float,
) -> tuple[float, bool, int]:
    after = scan_ticker["t"].gt(entry_t)
    before_end = (
        scan_ticker["t"].lt(end_t)
        if end_exclusive
        else scan_ticker["t"].le(end_t)
    )
    future = scan_ticker.loc[after & before_end].copy()
    if future.empty or entry_close <= 0:
        return 0.0, False, 0

    returns = (
        pd.to_numeric(future["c"], errors="coerce") / entry_close - 1.0
    ) * 100.0
    finite = returns.replace([np.inf, -np.inf], np.nan).dropna()
    peak = max(0.0, float(finite.max())) if len(finite) else 0.0
    runner = bool(
        future["runner_cross_now"].fillna(False).astype(bool).any()
    )
    return peak, runner, int(len(future))


def select_episode_entries(
    scan: pd.DataFrame,
    trace: pd.DataFrame,
    *,
    sample_tickers: int = SAMPLE_TICKERS_PER_DAY,
    max_episodes_per_ticker: int = MAX_EPISODES_PER_TICKER,
) -> pd.DataFrame:
    annotated = _annotate_scan(scan)
    trace_index = trace.set_index(["ticker", "t"], drop=False)
    scan_index = annotated.set_index(["ticker", "t"], drop=False)

    candidates: list[dict[str, object]] = []
    for episode in enumerate_watch_episodes(trace):
        key = (episode.ticker, episode.entry_t)
        if key not in scan_index.index or key not in trace_index.index:
            continue
        scan_row = scan_index.loc[key]
        trace_row = trace_index.loc[key]
        if isinstance(scan_row, pd.DataFrame) or isinstance(trace_row, pd.DataFrame):
            raise ValueError("episode entry key is not unique")
        if bool(scan_row["already_runner"]):
            continue
        item = scan_row.to_dict()
        item["attention_score"] = float(trace_row["attention_score"])
        item["attention_rank"] = trace_row["attention_rank"]
        item["episode_end_t"] = int(episode.end_t)
        item["episode_end_exclusive"] = bool(episode.end_exclusive)
        item["episode_ordinal"] = int(episode.ordinal)
        candidates.append(item)

    if not candidates:
        return pd.DataFrame()
    frame = pd.DataFrame(candidates)
    day = str(frame["trading_day"].iloc[0])
    selected_tickers = sorted(
        frame["ticker"].astype(str).unique(),
        key=lambda ticker: _selection_key(day, ticker),
    )[:sample_tickers]
    frame = frame.loc[frame["ticker"].isin(selected_tickers)].copy()
    frame = frame.sort_values(
        ["ticker", "t", "episode_ordinal"], kind="stable"
    )
    return (
        frame.groupby("ticker", sort=False)
        .head(max_episodes_per_ticker)
        .reset_index(drop=True)
    )


def build_day_episode_examples(
    scan: pd.DataFrame,
    trace: pd.DataFrame,
    client: MassiveClient,
    day: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = select_episode_entries(scan, trace)
    if selected.empty:
        return selected, {
            "trading_day": day.isoformat(),
            "selected_tickers": 0,
            "candidate_episodes": 0,
            "eligible_episodes": 0,
        }

    second_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker in sorted(selected["ticker"].astype(str).unique()):
        payload = client.second_bars_range(ticker, day, day, adjusted=False)
        second_by_ticker[ticker] = _second_frame(payload)

    scan_by_ticker = {
        ticker: group.sort_values("t", kind="stable")
        for ticker, group in scan.groupby("ticker", sort=False)
    }
    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        ticker = str(row.ticker)
        entry_t = int(row.t)
        entry_close = float(row.c)
        peak, runner, future_points = _episode_target(
            scan_by_ticker[ticker],
            entry_t=entry_t,
            end_t=int(row.episode_end_t),
            end_exclusive=bool(row.episode_end_exclusive),
            entry_close=entry_close,
        )
        item = row._asdict()
        item.update(second_path_features(second_by_ticker[ticker], entry_t))
        item["remaining_episode_peak_return_pct"] = peak
        item["runner_in_remaining_episode"] = runner
        item["episode_future_points"] = future_points
        item["episode_duration_minutes"] = max(
            (int(row.episode_end_t) - entry_t) / 60_000.0,
            0.0,
        )
        rows.append(item)

    examples = pd.DataFrame(rows)
    return examples, {
        "trading_day": day.isoformat(),
        "selected_tickers": int(selected["ticker"].nunique()),
        "candidate_episodes": int(len(selected)),
        "eligible_episodes": int(len(examples)),
        "second_rows": int(sum(len(frame) for frame in second_by_ticker.values())),
        "second_nonempty_tickers": int(
            sum(not frame.empty for frame in second_by_ticker.values())
        ),
    }


def _episode_target_summary(frame: pd.DataFrame) -> dict[str, object]:
    work = frame.rename(
        columns={
            "remaining_episode_peak_return_pct": "future_max_3m_return_pct"
        }
    )
    summary = _target_summary(work)
    duration = pd.to_numeric(
        frame["episode_duration_minutes"], errors="coerce"
    ).dropna()
    runner = frame["runner_in_remaining_episode"].fillna(False).astype(bool)
    summary.update(
        {
            "runner_rate": float(runner.mean()) if len(runner) else None,
            "duration_median_minutes": (
                float(duration.median()) if len(duration) else None
            ),
            "duration_p90_minutes": (
                float(duration.quantile(0.90)) if len(duration) else None
            ),
        }
    )
    return summary


def _episode_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, object]:
    renamed = frame.rename(
        columns={
            "remaining_episode_peak_return_pct": "future_max_3m_return_pct"
        }
    )
    return _metrics(renamed, prediction_column)


def fit_and_evaluate_episode(
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
    if len(fit) < 80 or len(evaluation) < 60:
        raise ValueError(
            f"insufficient episode rows: fit={len(fit)} eval={len(evaluation)}"
        )

    target = "remaining_episode_peak_return_pct"
    baseline = HistGradientBoostingRegressor(**MODEL_KWARGS)
    extended = HistGradientBoostingRegressor(**MODEL_KWARGS)
    baseline.fit(
        fit[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit[target],
    )
    extended.fit(
        fit[BASELINE_FEATURES + SECOND_FEATURES].replace(
            [np.inf, -np.inf], np.nan
        ),
        fit[target],
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
    pooled_baseline = _episode_metrics(evaluation, "baseline_prediction")
    pooled_extended = _episode_metrics(evaluation, "extended_prediction")

    by_day: dict[str, object] = {}
    day_differences: list[float] = []
    valid_day_comparisons = 0
    nonlower_days = 0
    minimum_rows = True
    for eval_day in eval_days:
        frame = evaluation.loc[
            evaluation["trading_day"].astype(str).eq(eval_day)
        ]
        b = _episode_metrics(frame, "baseline_prediction")
        e = _episode_metrics(frame, "extended_prediction")
        b_s = b.get("spearman")
        e_s = e.get("spearman")
        diff = None
        if b_s is not None and e_s is not None:
            diff = float(e_s - b_s)
            day_differences.append(diff)
            valid_day_comparisons += 1
            if e_s >= b_s:
                nonlower_days += 1
        if len(frame) < 15:
            minimum_rows = False
        by_day[eval_day] = {
            "target": _episode_target_summary(frame),
            "baseline": b,
            "extended": e,
            "incremental_spearman": diff,
        }

    pooled_b = pooled_baseline.get("spearman")
    pooled_e = pooled_extended.get("spearman")
    mean_day_increment = (
        float(np.mean(day_differences))
        if valid_day_comparisons == len(eval_days)
        else None
    )
    top_b = pooled_baseline.get("top_quartile_target_mean")
    top_e = pooled_extended.get("top_quartile_target_mean")
    mae_b = pooled_baseline.get("mae")
    mae_e = pooled_extended.get("mae")
    gate = bool(
        minimum_rows
        and pooled_b is not None
        and pooled_e is not None
        and pooled_e > pooled_b
        and mean_day_increment is not None
        and mean_day_increment >= 0
        and nonlower_days >= 3
        and mae_b is not None
        and mae_e is not None
        and mae_e <= mae_b
        and top_b is not None
        and top_e is not None
        and top_e >= top_b
    )

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
        "mean_day_incremental_spearman": mean_day_increment,
        "nonlower_eval_days": nonlower_days,
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
    pieces: list[pd.DataFrame] = []
    day_summaries: list[dict[str, object]] = []
    for day in daterange(start, end):
        if not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        trace = replay_attention(scan, AttentionConfig())
        examples, sample_summary = build_day_episode_examples(
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
        raise ValueError("watch-episode probe produced no examples")
    dataset = pd.concat(pieces, ignore_index=True)
    modeled, evaluation = fit_and_evaluate_episode(dataset)
    second_counts = pd.to_numeric(
        modeled["active_seconds_60"], errors="coerce"
    ).fillna(0)
    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sample_tickers_per_day": SAMPLE_TICKERS_PER_DAY,
        "max_episodes_per_ticker": MAX_EPISODES_PER_TICKER,
        "fit_end": FIT_END,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
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
        description="Evaluate one-second information on adaptive WATCH episodes."
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
        store, scan_client, second_client, args.start, args.end
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
