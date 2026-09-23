"""Diagnose intraminute observability of crossings missed by minute cadence."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange
from .second_path_attention_probe import SECOND_CACHE_DIR, SECOND_MS, _second_frame

EVAL_DAYS = [
    "2026-04-01",
    "2026-04-02",
    "2026-04-06",
    "2026-04-07",
    "2026-04-08",
]
RUNNER_THRESHOLD_PCT = 10.0


def crossing_observability_rows(scan: pd.DataFrame) -> pd.DataFrame:
    ordered = scan.sort_values(
        ["trading_day", "ticker", "t"], kind="stable"
    ).copy()
    groups = ordered.groupby(["trading_day", "ticker"], sort=False)
    ordered["previous_bar_t"] = groups["t"].shift(1)
    crossings = ordered.loc[
        ordered["runner_cross_now"].fillna(False).astype(bool)
    ].copy()
    gap = (
        pd.to_numeric(crossings["t"], errors="coerce")
        - pd.to_numeric(crossings["previous_bar_t"], errors="coerce")
    )
    crossings["prior_gap_ms"] = gap
    crossings["has_exact_prior_minute"] = gap.eq(MINUTE_MS)
    crossings["no_prior_bar"] = crossings["previous_bar_t"].isna()
    return crossings


def _gap_bucket(row: pd.Series) -> str:
    if bool(row["no_prior_bar"]):
        return "no_prior_bar"
    gap = float(row["prior_gap_ms"]) / MINUTE_MS
    if gap <= 2:
        return "gap_2m"
    if gap <= 5:
        return "gap_3_5m"
    if gap <= 30:
        return "gap_6_30m"
    return "gap_gt_30m"


def add_second_observability(
    blind: pd.DataFrame,
    second_client: MassiveClient,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}

    for row in blind.itertuples(index=False):
        day_text = str(row.trading_day)
        ticker = str(row.ticker)
        key = (day_text, ticker)
        if key not in cache:
            day = date.fromisoformat(day_text)
            payload = second_client.second_bars_range(
                ticker, day, day, adjusted=False
            )
            cache[key] = _second_frame(payload)
        seconds = cache[key]
        decision_t = int(row.t)
        window = seconds.loc[
            seconds["t"].ge(decision_t - MINUTE_MS)
            & (seconds["t"] + SECOND_MS).le(decision_t)
        ].copy()

        previous_close = float(row.previous_close)
        threshold_price = previous_close * (1.0 + RUNNER_THRESHOLD_PCT / 100.0)
        item = {
            "trading_day": day_text,
            "ticker": ticker,
            "crossing_t": decision_t,
            "crossing_return_pct": float(row.return_from_previous_close_pct),
            "previous_close": previous_close,
            "gap_bucket": _gap_bucket(pd.Series(row._asdict())),
            "second_rows_in_crossing_minute": int(len(window)),
            "second_data_present": bool(len(window)),
            "second_threshold_found": False,
            "first_active_second_already_crossed": False,
            "active_seconds_before_cross": None,
            "wall_seconds_from_first_activity_to_cross": None,
            "pre_cross_peak_return_pct": None,
            "first_active_return_pct": None,
        }
        if window.empty:
            rows.append(item)
            continue

        returns = (
            pd.to_numeric(window["c"], errors="coerce") / previous_close - 1.0
        ) * 100.0
        item["first_active_return_pct"] = float(returns.iloc[0])
        crossed = pd.to_numeric(window["c"], errors="coerce").ge(threshold_price)
        if not crossed.any():
            rows.append(item)
            continue

        first_cross_index = crossed[crossed].index[0]
        first_cross_t = int(window.loc[first_cross_index, "t"])
        first_active_t = int(window.iloc[0]["t"])
        before = window.loc[window["t"].lt(first_cross_t)]
        item["second_threshold_found"] = True
        item["first_active_second_already_crossed"] = first_cross_t == first_active_t
        item["active_seconds_before_cross"] = int(len(before))
        item["wall_seconds_from_first_activity_to_cross"] = float(
            (first_cross_t - first_active_t) / 1_000.0
        )
        if len(before):
            pre_returns = (
                pd.to_numeric(before["c"], errors="coerce") / previous_close - 1.0
            ) * 100.0
            item["pre_cross_peak_return_pct"] = float(pre_returns.max())
        rows.append(item)

    frame = pd.DataFrame(rows)
    return frame, {
        "unique_ticker_days_requested": int(len(cache)),
        "second_client_stats": second_client.stats.to_dict(),
    }


def _numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
    }


def summarize(
    crossings: pd.DataFrame,
    second_rows: pd.DataFrame,
) -> dict[str, object]:
    total = int(len(crossings))
    exact = int(crossings["has_exact_prior_minute"].sum())
    blind = total - exact

    present = second_rows["second_data_present"].fillna(False).astype(bool)
    found = second_rows["second_threshold_found"].fillna(False).astype(bool)
    already = second_rows[
        "first_active_second_already_crossed"
    ].fillna(False).astype(bool)
    pre_seconds = pd.to_numeric(
        second_rows["active_seconds_before_cross"], errors="coerce"
    )

    by_gap: dict[str, object] = {}
    for bucket, group in second_rows.groupby("gap_bucket", sort=True):
        group_found = group["second_threshold_found"].fillna(False).astype(bool)
        group_pre = pd.to_numeric(
            group["active_seconds_before_cross"], errors="coerce"
        )
        by_gap[str(bucket)] = {
            "crossings": int(len(group)),
            "second_data_coverage": float(
                group["second_data_present"].fillna(False).astype(bool).mean()
            ),
            "threshold_found_rate": float(group_found.mean()),
            "pre_cross_seconds": _numeric_summary(group_pre),
        }

    by_day: dict[str, object] = {}
    for day in EVAL_DAYS:
        cross_day = crossings.loc[crossings["trading_day"].astype(str).eq(day)]
        second_day = second_rows.loc[
            second_rows["trading_day"].astype(str).eq(day)
        ]
        day_found = second_day[
            "second_threshold_found"
        ].fillna(False).astype(bool)
        day_pre = pd.to_numeric(
            second_day["active_seconds_before_cross"], errors="coerce"
        )
        by_day[day] = {
            "crossings": int(len(cross_day)),
            "exact_prior_minute_count": int(
                cross_day["has_exact_prior_minute"].sum()
            ),
            "blind_crossings": int(
                (~cross_day["has_exact_prior_minute"]).sum()
            ),
            "second_data_coverage": (
                float(
                    second_day["second_data_present"]
                    .fillna(False)
                    .astype(bool)
                    .mean()
                )
                if len(second_day)
                else None
            ),
            "threshold_found_rate": (
                float(day_found.mean()) if len(second_day) else None
            ),
            "pre_cross_seconds": _numeric_summary(day_pre),
        }

    denominator = int(found.sum())
    return {
        "crossings": total,
        "exact_prior_minute_count": exact,
        "exact_prior_minute_rate": float(exact / total) if total else None,
        "blind_crossings": blind,
        "second_data_coverage_of_blind": float(present.mean()) if blind else None,
        "second_threshold_found_count": denominator,
        "second_threshold_found_rate": float(found.mean()) if blind else None,
        "first_active_second_already_crossed_count": int((already & found).sum()),
        "first_active_second_already_crossed_rate": (
            float((already & found).sum() / denominator)
            if denominator
            else None
        ),
        "has_pre_cross_second_count": int(
            ((pre_seconds > 0) & found).sum()
        ),
        "has_pre_cross_second_rate": (
            float(((pre_seconds > 0) & found).sum() / denominator)
            if denominator
            else None
        ),
        "pre_cross_seconds_ge_5_rate": (
            float(((pre_seconds >= 5) & found).sum() / denominator)
            if denominator
            else None
        ),
        "pre_cross_seconds_ge_10_rate": (
            float(((pre_seconds >= 10) & found).sum() / denominator)
            if denominator
            else None
        ),
        "pre_cross_seconds_ge_30_rate": (
            float(((pre_seconds >= 30) & found).sum() / denominator)
            if denominator
            else None
        ),
        "active_seconds_before_cross": _numeric_summary(pre_seconds),
        "wall_seconds_from_first_activity_to_cross": _numeric_summary(
            second_rows["wall_seconds_from_first_activity_to_cross"]
        ),
        "pre_cross_peak_return_pct": _numeric_summary(
            second_rows["pre_cross_peak_return_pct"]
        ),
        "first_active_return_pct": _numeric_summary(
            second_rows["first_active_return_pct"]
        ),
        "by_gap": by_gap,
        "by_day": by_day,
    }


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    crossings: list[pd.DataFrame] = []
    days: list[dict[str, object]] = []
    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in EVAL_DAYS or not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        day_crossings = crossing_observability_rows(scan)
        crossings.append(day_crossings)
        days.append(
            {
                "trading_day": day_text,
                "scan_rows": int(len(scan)),
                "runner_crossings": int(len(day_crossings)),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )

    if not crossings:
        raise ValueError("no opened evaluation crossings found")
    crossing_frame = pd.concat(crossings, ignore_index=True)
    found_days = sorted(crossing_frame["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")

    blind = crossing_frame.loc[
        ~crossing_frame["has_exact_prior_minute"]
    ].copy()
    second_rows, second_audit = add_second_observability(blind, second_client)
    evaluation = summarize(crossing_frame, second_rows)
    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "eval_days": EVAL_DAYS,
        "runner_threshold_pct": RUNNER_THRESHOLD_PCT,
        "days": days,
        "evaluation": evaluation,
        "second_audit": second_audit,
        "flatfile_stats": store.stats.to_dict(),
    }
    return second_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose one-second observability of minute blind spots."
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key), FLATFILE_CACHE_DIR
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
    output, summary = run_probe(
        store, scan_client, second_client, args.start, args.end
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
