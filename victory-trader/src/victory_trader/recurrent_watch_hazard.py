"""Recurrent next-step runner hazard for WATCH attention."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS, replay_attention
from .attention_runtime import AttentionConfig
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange
from .second_path_attention_probe import BASELINE_FEATURES, _annotate_scan

FIT_END = "2026-01-26"
EVAL_DAYS = [
    "2026-01-27",
    "2026-01-28",
    "2026-01-29",
    "2026-01-30",
    "2026-02-02",
]
STATE_FEATURES = ["focus_age_minutes", "watch_run_age_minutes"]
MODEL_FEATURES = [*BASELINE_FEATURES, *STATE_FEATURES]
MODEL_KWARGS = {
    "learning_rate": 0.05,
    "max_iter": 120,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": 17,
}


def _state_history(trace: pd.DataFrame) -> pd.DataFrame:
    work = trace.sort_values(["ticker", "t"], kind="stable").copy()
    state = work["state"].astype(str)
    focus = state.isin({"watch", "hot"})
    watch = state.eq("watch")

    grouped_t = work.groupby("ticker", sort=False)["t"]
    previous_t = grouped_t.shift(1)
    gap = pd.to_numeric(work["t"], errors="coerce") - pd.to_numeric(
        previous_t, errors="coerce"
    )
    consecutive = gap.eq(MINUTE_MS)

    previous_focus = focus.groupby(work["ticker"], sort=False).shift(
        1, fill_value=False
    )
    focus_start = focus & (~previous_focus | ~consecutive)
    focus_group = focus_start.groupby(work["ticker"], sort=False).cumsum()
    focus_ordinal = (
        work.loc[focus]
        .groupby(
            [work.loc[focus, "ticker"], focus_group.loc[focus]],
            sort=False,
        )
        .cumcount()
        .astype(float)
    )
    work["focus_age_minutes"] = np.nan
    work.loc[focus, "focus_age_minutes"] = focus_ordinal + 1.0

    previous_watch = watch.groupby(work["ticker"], sort=False).shift(
        1, fill_value=False
    )
    watch_start = watch & (~previous_watch | ~consecutive)
    watch_group = watch_start.groupby(work["ticker"], sort=False).cumsum()
    watch_ordinal = (
        work.loc[watch]
        .groupby(
            [work.loc[watch, "ticker"], watch_group.loc[watch]],
            sort=False,
        )
        .cumcount()
        .astype(float)
    )
    work["watch_run_age_minutes"] = np.nan
    work.loc[watch, "watch_run_age_minutes"] = watch_ordinal + 1.0

    return work


def build_day_watch_rows(
    scan: pd.DataFrame,
    trace: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    annotated = _annotate_scan(scan).sort_values(
        ["ticker", "t"], kind="stable"
    )
    next_t = annotated.groupby("ticker", sort=False)["t"].shift(-1)
    next_cross = (
        annotated.groupby("ticker", sort=False)["runner_cross_now"]
        .shift(-1)
        .astype("boolean")
    )
    annotated["has_exact_next_minute"] = (
        pd.to_numeric(next_t, errors="coerce")
        - pd.to_numeric(annotated["t"], errors="coerce")
    ).eq(MINUTE_MS)
    annotated["next_step_runner_cross"] = (
        next_cross.fillna(False).astype(bool)
        & annotated["has_exact_next_minute"]
    )

    state = _state_history(trace)
    watch = state.loc[state["state"].astype(str).eq("watch")].copy()
    merged = watch.merge(
        annotated,
        on=["trading_day", "ticker", "t"],
        how="inner",
        suffixes=("", "_scan"),
        validate="one_to_one",
    )
    merged = merged.loc[
        ~merged["already_runner"].fillna(True)
        & merged["has_exact_next_minute"].fillna(False)
    ].copy()
    merged["target_next_cross"] = (
        merged["next_step_runner_cross"].fillna(False).astype(int)
    )
    return merged, {
        "watch_rows": int(len(merged)),
        "positive_next_crossings": int(merged["target_next_cross"].sum()),
        "watch_tickers": int(merged["ticker"].nunique()) if len(merged) else 0,
    }


def _safe_auc(y: pd.Series, score: pd.Series) -> float | None:
    y_arr = pd.to_numeric(y, errors="coerce")
    s_arr = pd.to_numeric(score, errors="coerce")
    valid = y_arr.notna() & s_arr.notna()
    y_arr = y_arr.loc[valid].astype(int)
    s_arr = s_arr.loc[valid].astype(float)
    if y_arr.nunique() < 2:
        return None
    return float(roc_auc_score(y_arr, s_arr))


def _safe_ap(y: pd.Series, score: pd.Series) -> float | None:
    y_arr = pd.to_numeric(y, errors="coerce")
    s_arr = pd.to_numeric(score, errors="coerce")
    valid = y_arr.notna() & s_arr.notna()
    y_arr = y_arr.loc[valid].astype(int)
    s_arr = s_arr.loc[valid].astype(float)
    if int(y_arr.sum()) == 0:
        return None
    return float(average_precision_score(y_arr, s_arr))


def _top_budget_metrics(
    frame: pd.DataFrame,
    score_column: str,
    *,
    budget: int = 10,
) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "selected_rows": 0,
            "positive_events": 0,
            "captured_events": 0,
            "capture_rate": None,
            "precision": None,
            "mean_positive_rank": None,
        }

    selected_indices: list[int] = []
    positive_ranks: list[float] = []
    for _, group in frame.groupby(
        ["trading_day", "t"], sort=False, dropna=False
    ):
        ordered = group.sort_values(
            [score_column, "ticker"],
            ascending=[False, True],
            kind="stable",
        )
        selected_indices.extend(ordered.head(budget).index.tolist())
        positive = ordered["target_next_cross"].astype(bool).to_numpy()
        if positive.any():
            ranks = np.arange(1, len(ordered) + 1, dtype=float)
            positive_ranks.extend(ranks[positive].tolist())

    selected = frame.loc[selected_indices]
    positive_events = int(frame["target_next_cross"].sum())
    captured = int(selected["target_next_cross"].sum())
    return {
        "selected_rows": int(len(selected)),
        "positive_events": positive_events,
        "captured_events": captured,
        "capture_rate": (
            float(captured / positive_events) if positive_events else None
        ),
        "precision": (
            float(captured / len(selected)) if len(selected) else None
        ),
        "mean_positive_rank": (
            float(np.mean(positive_ranks)) if positive_ranks else None
        ),
    }


def _score_metrics(
    frame: pd.DataFrame,
    score_column: str,
    *,
    probability: bool,
) -> dict[str, object]:
    y = frame["target_next_cross"].astype(int)
    score = pd.to_numeric(frame[score_column], errors="coerce")
    valid = score.notna()
    yv = y.loc[valid]
    sv = score.loc[valid]
    result: dict[str, object] = {
        "rows": int(len(yv)),
        "positives": int(yv.sum()),
        "positive_rate": float(yv.mean()) if len(yv) else None,
        "average_precision": _safe_ap(yv, sv),
        "roc_auc": _safe_auc(yv, sv),
        "top10": _top_budget_metrics(
            frame.loc[valid], score_column, budget=10
        ),
    }
    result["brier"] = (
        float(brier_score_loss(yv, sv))
        if probability and len(yv)
        else None
    )
    return result


def fit_and_evaluate(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    work = dataset.copy()
    day = work["trading_day"].astype(str)
    is_eval = day.isin(EVAL_DAYS)
    is_fit = day.le(FIT_END)
    if (~(is_fit | is_eval)).any():
        bad = sorted(day.loc[~(is_fit | is_eval)].unique().tolist())
        raise ValueError(f"rows outside frozen fit/eval chronology: {bad}")

    work["split"] = np.where(is_eval, "eval", "fit")
    fit = work.loc[work["split"].eq("fit")].copy()
    evaluation = work.loc[work["split"].eq("eval")].copy()
    found_eval = sorted(evaluation["trading_day"].astype(str).unique())
    if found_eval != EVAL_DAYS:
        raise ValueError(
            f"expected evaluation sessions {EVAL_DAYS}, found {found_eval}"
        )
    if int(fit["target_next_cross"].sum()) == 0:
        raise ValueError("fit set has no positive next-step crossings")

    classifier = HistGradientBoostingClassifier(**MODEL_KWARGS)
    classifier.fit(
        fit[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["target_next_cross"].astype(int),
    )
    work["hazard_probability"] = classifier.predict_proba(
        work[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]

    fit_positive_rate = float(fit["target_next_cross"].mean())
    work["climatology_probability"] = fit_positive_rate

    evaluation = work.loc[work["split"].eq("eval")].copy()
    baseline = _score_metrics(
        evaluation, "attention_score", probability=False
    )
    learned = _score_metrics(
        evaluation, "hazard_probability", probability=True
    )
    climatology_brier = float(
        brier_score_loss(
            evaluation["target_next_cross"].astype(int),
            evaluation["climatology_probability"],
        )
    )

    by_day: dict[str, object] = {}
    nonlower_capture_days = 0
    minimum_support = True
    for eval_day in EVAL_DAYS:
        frame = evaluation.loc[
            evaluation["trading_day"].astype(str).eq(eval_day)
        ]
        b = _score_metrics(frame, "attention_score", probability=False)
        l = _score_metrics(frame, "hazard_probability", probability=True)
        b_capture = b["top10"]["capture_rate"]
        l_capture = l["top10"]["capture_rate"]
        if (
            b_capture is not None
            and l_capture is not None
            and l_capture >= b_capture
        ):
            nonlower_capture_days += 1
        if len(frame) < 1000 or int(frame["target_next_cross"].sum()) < 10:
            minimum_support = False
        by_day[eval_day] = {
            "rows": int(len(frame)),
            "positives": int(frame["target_next_cross"].sum()),
            "baseline": b,
            "learned": l,
        }

    baseline_ap = baseline.get("average_precision")
    learned_ap = learned.get("average_precision")
    baseline_capture = baseline["top10"]["capture_rate"]
    learned_capture = learned["top10"]["capture_rate"]
    baseline_rank = baseline["top10"]["mean_positive_rank"]
    learned_rank = learned["top10"]["mean_positive_rank"]
    learned_brier = learned.get("brier")

    gate = bool(
        minimum_support
        and baseline_ap is not None
        and learned_ap is not None
        and learned_ap > baseline_ap
        and baseline_capture is not None
        and learned_capture is not None
        and learned_capture > baseline_capture
        and nonlower_capture_days >= 4
        and learned_brier is not None
        and learned_brier < climatology_brier
        and baseline_rank is not None
        and learned_rank is not None
        and learned_rank < baseline_rank
    )

    return work, {
        "fit_days": sorted(fit["trading_day"].astype(str).unique()),
        "eval_days": EVAL_DAYS,
        "fit_rows": int(len(fit)),
        "fit_positives": int(fit["target_next_cross"].sum()),
        "fit_positive_rate": fit_positive_rate,
        "eval_rows": int(len(evaluation)),
        "eval_positives": int(evaluation["target_next_cross"].sum()),
        "baseline": baseline,
        "learned": learned,
        "climatology_brier": climatology_brier,
        "nonlower_capture_days": nonlower_capture_days,
        "by_day": by_day,
        "promotion_gate_pass": gate,
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
        rows, sample = build_day_watch_rows(scan, trace)
        sample["trading_day"] = day.isoformat()
        sample["scan_rows"] = int(len(scan))
        sample["trace_rows"] = int(len(trace))
        sample["point_in_time_common_stocks"] = scan_summary.get(
            "point_in_time_common_stocks"
        )
        day_summaries.append(sample)
        if not rows.empty:
            pieces.append(rows)

    if not pieces:
        raise ValueError("recurrent WATCH hazard produced no rows")

    dataset = pd.concat(pieces, ignore_index=True)
    modeled, evaluation = fit_and_evaluate(dataset)
    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fit_end": FIT_END,
        "eval_days": EVAL_DAYS,
        "population": "all causal pre-runner WATCH decisions with exact next minute",
        "target": "first +10pct prior-close crossing at next minute decision",
        "features": MODEL_FEATURES,
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
        description="Evaluate recurrent next-step runner hazard for WATCH rows."
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
