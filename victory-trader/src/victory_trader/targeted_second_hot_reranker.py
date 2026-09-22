"""Targeted one-second reranking inside a frozen minute-hazard shortlist."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import replay_attention
from .attention_runtime import AttentionConfig
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange
from .recurrent_watch_hazard import (
    MODEL_FEATURES,
    MODEL_KWARGS,
    build_day_watch_rows,
)
from .second_path_attention_probe import (
    SECOND_CACHE_DIR,
    SECOND_FEATURES,
    _second_frame,
    second_path_features,
)

STAGE1_FIT_END = "2026-01-16"
STAGE2_FIT_START = "2026-01-20"
STAGE2_FIT_END = "2026-02-02"
EVAL_DAYS = [
    "2026-02-03",
    "2026-02-04",
    "2026-02-05",
    "2026-02-06",
    "2026-02-09",
]
SHORTLIST_BUDGET = 20
HOT_BUDGET = 10
MINUTE_RERANK_FEATURES = [*MODEL_FEATURES, "stage1_hazard_probability"]
SECOND_RERANK_FEATURES = [*MINUTE_RERANK_FEATURES, *SECOND_FEATURES]


def _safe_ap(y: pd.Series, score: pd.Series) -> float | None:
    yv = pd.to_numeric(y, errors="coerce")
    sv = pd.to_numeric(score, errors="coerce")
    valid = yv.notna() & sv.notna()
    yv = yv.loc[valid].astype(int)
    sv = sv.loc[valid].astype(float)
    if int(yv.sum()) == 0:
        return None
    return float(average_precision_score(yv, sv))


def _safe_roc(y: pd.Series, score: pd.Series) -> float | None:
    yv = pd.to_numeric(y, errors="coerce")
    sv = pd.to_numeric(score, errors="coerce")
    valid = yv.notna() & sv.notna()
    yv = yv.loc[valid].astype(int)
    sv = sv.loc[valid].astype(float)
    if yv.nunique() < 2:
        return None
    return float(roc_auc_score(yv, sv))


def fit_stage1(all_rows: pd.DataFrame) -> HistGradientBoostingClassifier:
    fit = all_rows.loc[
        all_rows["trading_day"].astype(str).le(STAGE1_FIT_END)
    ].copy()
    if fit.empty or int(fit["target_next_cross"].sum()) == 0:
        raise ValueError("stage-1 fit set is empty or has no positives")
    model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    model.fit(
        fit[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["target_next_cross"].astype(int),
    )
    return model


def add_stage1_scores(
    rows: pd.DataFrame,
    model: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    result = rows.copy()
    result["stage1_hazard_probability"] = model.predict_proba(
        result[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    return result


def shortlist_rows(
    rows: pd.DataFrame,
    *,
    budget: int = SHORTLIST_BUDGET,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in rows.groupby(["trading_day", "t"], sort=False):
        ordered = group.sort_values(
            ["stage1_hazard_probability", "ticker"],
            ascending=[False, True],
            kind="stable",
        )
        pieces.append(ordered.head(budget))
    if not pieces:
        return rows.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def add_second_features(
    candidates: pd.DataFrame,
    second_client: MassiveClient,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pieces: list[pd.DataFrame] = []
    request_ticker_days = 0
    nonempty_ticker_days = 0
    second_rows = 0

    for day, day_rows in candidates.groupby("trading_day", sort=True):
        day_date = date.fromisoformat(str(day))
        seconds_by_ticker: dict[str, pd.DataFrame] = {}
        tickers = sorted(day_rows["ticker"].astype(str).unique())
        for ticker in tickers:
            request_ticker_days += 1
            payload = second_client.second_bars_range(
                ticker, day_date, day_date, adjusted=False
            )
            second_frame = _second_frame(payload)
            seconds_by_ticker[ticker] = second_frame
            second_rows += len(second_frame)
            if not second_frame.empty:
                nonempty_ticker_days += 1

        enriched_rows: list[dict[str, object]] = []
        for row in day_rows.itertuples(index=False):
            item = row._asdict()
            item.update(
                second_path_features(
                    seconds_by_ticker[str(row.ticker)],
                    int(row.t),
                )
            )
            enriched_rows.append(item)
        pieces.append(pd.DataFrame(enriched_rows))

    enriched = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else candidates.iloc[0:0].copy()
    )
    return enriched, {
        "ticker_day_requests": request_ticker_days,
        "nonempty_ticker_days": nonempty_ticker_days,
        "second_rows": second_rows,
        "row_coverage": (
            float(
                pd.to_numeric(
                    enriched.get("active_seconds_60"), errors="coerce"
                )
                .fillna(0)
                .gt(0)
                .mean()
            )
            if len(enriched)
            else None
        ),
    }


def fit_stage2_models(
    candidates: pd.DataFrame,
) -> tuple[
    HistGradientBoostingClassifier,
    HistGradientBoostingClassifier,
]:
    day = candidates["trading_day"].astype(str)
    fit = candidates.loc[
        day.ge(STAGE2_FIT_START) & day.le(STAGE2_FIT_END)
    ].copy()
    if fit.empty or int(fit["target_next_cross"].sum()) == 0:
        raise ValueError("stage-2 fit set is empty or has no positives")

    minute_model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    second_model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    minute_model.fit(
        fit[MINUTE_RERANK_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["target_next_cross"].astype(int),
    )
    second_model.fit(
        fit[SECOND_RERANK_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["target_next_cross"].astype(int),
    )
    return minute_model, second_model


def add_stage2_scores(
    candidates: pd.DataFrame,
    minute_model: HistGradientBoostingClassifier,
    second_model: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    result = candidates.copy()
    result["minute_rerank_probability"] = minute_model.predict_proba(
        result[MINUTE_RERANK_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    result["second_rerank_probability"] = second_model.predict_proba(
        result[SECOND_RERANK_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    return result


def _hot_budget_metrics(
    candidates: pd.DataFrame,
    all_watch: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    selected_indices: list[int] = []
    positive_ranks: list[float] = []

    for _, group in candidates.groupby(["trading_day", "t"], sort=False):
        ordered = group.sort_values(
            [score_column, "ticker"],
            ascending=[False, True],
            kind="stable",
        )
        selected_indices.extend(ordered.head(HOT_BUDGET).index.tolist())
        positive = ordered["target_next_cross"].astype(bool).to_numpy()
        if positive.any():
            ranks = np.arange(1, len(ordered) + 1, dtype=float)
            positive_ranks.extend(ranks[positive].tolist())

    selected = candidates.loc[selected_indices]
    all_positive = int(all_watch["target_next_cross"].sum())
    captured = int(selected["target_next_cross"].sum())
    return {
        "selected_rows": int(len(selected)),
        "all_watch_positive_events": all_positive,
        "captured_events": captured,
        "all_watch_capture_rate": (
            float(captured / all_positive) if all_positive else None
        ),
        "precision": (
            float(captured / len(selected)) if len(selected) else None
        ),
        "mean_positive_candidate_rank": (
            float(np.mean(positive_ranks)) if positive_ranks else None
        ),
    }


def _reranker_metrics(
    candidates: pd.DataFrame,
    all_watch: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    y = candidates["target_next_cross"].astype(int)
    score = pd.to_numeric(candidates[score_column], errors="coerce")
    valid = score.notna()
    yv = y.loc[valid]
    sv = score.loc[valid]
    return {
        "candidate_rows": int(len(yv)),
        "candidate_positives": int(yv.sum()),
        "average_precision": _safe_ap(yv, sv),
        "roc_auc": _safe_roc(yv, sv),
        "brier": (
            float(brier_score_loss(yv, sv)) if len(yv) else None
        ),
        "hot10": _hot_budget_metrics(
            candidates.loc[valid], all_watch, score_column
        ),
    }


def evaluate(
    scored_candidates: pd.DataFrame,
    all_watch: pd.DataFrame,
) -> dict[str, object]:
    eval_candidates = scored_candidates.loc[
        scored_candidates["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    eval_watch = all_watch.loc[
        all_watch["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()

    found_days = sorted(eval_watch["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")

    all_positive = int(eval_watch["target_next_cross"].sum())
    candidate_positive = int(eval_candidates["target_next_cross"].sum())
    shortlist_coverage = (
        float(candidate_positive / all_positive) if all_positive else None
    )

    minute = _reranker_metrics(
        eval_candidates, eval_watch, "minute_rerank_probability"
    )
    second = _reranker_metrics(
        eval_candidates, eval_watch, "second_rerank_probability"
    )

    by_day: dict[str, object] = {}
    nonlower_days = 0
    support_ok = True
    for eval_day in EVAL_DAYS:
        day_watch = eval_watch.loc[
            eval_watch["trading_day"].astype(str).eq(eval_day)
        ]
        day_candidates = eval_candidates.loc[
            eval_candidates["trading_day"].astype(str).eq(eval_day)
        ]
        minute_day = _reranker_metrics(
            day_candidates, day_watch, "minute_rerank_probability"
        )
        second_day = _reranker_metrics(
            day_candidates, day_watch, "second_rerank_probability"
        )
        minute_capture = minute_day["hot10"]["all_watch_capture_rate"]
        second_capture = second_day["hot10"]["all_watch_capture_rate"]
        if (
            minute_capture is not None
            and second_capture is not None
            and second_capture >= minute_capture
        ):
            nonlower_days += 1
        if int(day_watch["target_next_cross"].sum()) < 10:
            support_ok = False
        by_day[eval_day] = {
            "all_watch_rows": int(len(day_watch)),
            "all_watch_positives": int(day_watch["target_next_cross"].sum()),
            "candidate_rows": int(len(day_candidates)),
            "candidate_positives": int(
                day_candidates["target_next_cross"].sum()
            ),
            "shortlist_positive_coverage": (
                float(
                    day_candidates["target_next_cross"].sum()
                    / day_watch["target_next_cross"].sum()
                )
                if int(day_watch["target_next_cross"].sum()) > 0
                else None
            ),
            "minute": minute_day,
            "second": second_day,
        }

    minute_ap = minute.get("average_precision")
    second_ap = second.get("average_precision")
    minute_capture = minute["hot10"]["all_watch_capture_rate"]
    second_capture = second["hot10"]["all_watch_capture_rate"]
    minute_brier = minute.get("brier")
    second_brier = second.get("brier")
    minute_rank = minute["hot10"]["mean_positive_candidate_rank"]
    second_rank = second["hot10"]["mean_positive_candidate_rank"]

    gate = bool(
        support_ok
        and shortlist_coverage is not None
        and shortlist_coverage >= 0.95
        and minute_ap is not None
        and second_ap is not None
        and second_ap > minute_ap
        and minute_capture is not None
        and second_capture is not None
        and second_capture >= minute_capture
        and nonlower_days >= 4
        and minute_brier is not None
        and second_brier is not None
        and second_brier <= minute_brier
        and minute_rank is not None
        and second_rank is not None
        and second_rank < minute_rank
    )

    return {
        "eval_days": EVAL_DAYS,
        "all_watch_rows": int(len(eval_watch)),
        "all_watch_positives": all_positive,
        "candidate_rows": int(len(eval_candidates)),
        "candidate_positives": candidate_positive,
        "shortlist_positive_coverage": shortlist_coverage,
        "minute": minute,
        "second": second,
        "nonlower_capture_days": nonlower_days,
        "by_day": by_day,
        "promotion_gate_pass": gate,
    }


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    all_pieces: list[pd.DataFrame] = []
    day_summaries: list[dict[str, object]] = []

    for day in daterange(start, end):
        if not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        trace = replay_attention(scan, AttentionConfig())
        rows, summary = build_day_watch_rows(scan, trace)
        summary["trading_day"] = day.isoformat()
        summary["scan_rows"] = int(len(scan))
        summary["trace_rows"] = int(len(trace))
        summary["point_in_time_common_stocks"] = scan_summary.get(
            "point_in_time_common_stocks"
        )
        day_summaries.append(summary)
        if not rows.empty:
            all_pieces.append(rows)

    if not all_pieces:
        raise ValueError("targeted second reranker produced no WATCH rows")

    all_watch = pd.concat(all_pieces, ignore_index=True)
    stage1 = fit_stage1(all_watch)
    scored_watch = add_stage1_scores(all_watch, stage1)

    day_text = scored_watch["trading_day"].astype(str)
    rerank_period = day_text.ge(STAGE2_FIT_START) & (
        day_text.le(STAGE2_FIT_END) | day_text.isin(EVAL_DAYS)
    )
    candidates = shortlist_rows(
        scored_watch.loc[rerank_period].copy(),
        budget=SHORTLIST_BUDGET,
    )
    candidates, second_audit = add_second_features(
        candidates, second_client
    )
    minute_model, second_model = fit_stage2_models(candidates)
    scored_candidates = add_stage2_scores(
        candidates, minute_model, second_model
    )
    evaluation = evaluate(scored_candidates, scored_watch)

    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "stage1_fit_end": STAGE1_FIT_END,
        "stage2_fit_start": STAGE2_FIT_START,
        "stage2_fit_end": STAGE2_FIT_END,
        "eval_days": EVAL_DAYS,
        "shortlist_budget": SHORTLIST_BUDGET,
        "hot_budget": HOT_BUDGET,
        "stage1_features": MODEL_FEATURES,
        "minute_rerank_features": MINUTE_RERANK_FEATURES,
        "second_features": SECOND_FEATURES,
        "model": MODEL_KWARGS,
        "days": day_summaries,
        "all_watch_rows": int(len(scored_watch)),
        "candidate_rows": int(len(scored_candidates)),
        "candidate_tickers": int(scored_candidates["ticker"].nunique()),
        "second_data_audit": second_audit,
        "second_client_stats": second_client.stats.to_dict(),
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    return scored_candidates, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate targeted one-second reranking for HOT-10 allocation."
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
