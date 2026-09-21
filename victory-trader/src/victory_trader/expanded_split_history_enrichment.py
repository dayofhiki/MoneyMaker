from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .expanded_execution_feasible_hurdle_ev import execution_feasible_mask
from .massive_client import MassiveClient


LOOKBACK_DAYS = 730
RAW_SPLIT_COLUMNS = (
    "split_any_730d",
    "reverse_split_count_730d",
    "reverse_split_count_365d",
    "forward_split_count_730d",
    "stock_dividend_count_730d",
    "split_days_since_latest_reverse",
    "split_latest_reverse_consolidation",
    "split_max_reverse_consolidation",
)


def _positive(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or float(numeric) <= 0:
        return np.nan
    return float(numeric)


def _consolidation(record: dict[str, object]) -> float:
    before = _positive(record.get("split_from"))
    after = _positive(record.get("split_to"))
    if not np.isfinite(before) or not np.isfinite(after):
        return np.nan
    return before / after


def _prepare_events(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_ticker: dict[str, list[dict[str, object]]] = {}
    for raw in records:
        ticker = str(raw.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        try:
            execution_date = pd.Timestamp(
                str(raw.get("execution_date"))
            ).date()
        except Exception:
            continue
        item = dict(raw)
        item["_execution_date"] = execution_date
        by_ticker.setdefault(ticker, []).append(item)

    for ticker in by_ticker:
        by_ticker[ticker].sort(key=lambda row: row["_execution_date"])
    return by_ticker


def enrich_frame(
    frame: pd.DataFrame,
    events_by_ticker: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    result = frame.copy()
    result["split_history_query_success"] = 0.0
    result["split_history_strict_prior"] = 0.0
    for column in RAW_SPLIT_COLUMNS:
        result[column] = np.nan

    feasible = execution_feasible_mask(result)
    positions = np.flatnonzero(feasible.to_numpy(dtype=bool))

    for pos in positions:
        row = result.iloc[pos]
        trading_day = pd.Timestamp(str(row["trading_day"])).date()
        start = trading_day - timedelta(days=LOOKBACK_DAYS)
        ticker = str(row["ticker"]).upper()
        records = [
            item
            for item in events_by_ticker.get(ticker, [])
            if start <= item["_execution_date"] < trading_day
        ]

        strict_prior = all(
            item["_execution_date"] < trading_day for item in records
        )
        reverse = [
            item
            for item in records
            if str(item.get("adjustment_type", "")).lower()
            == "reverse_split"
        ]
        reverse_365 = [
            item
            for item in reverse
            if (trading_day - item["_execution_date"]).days <= 365
        ]
        forward = [
            item
            for item in records
            if str(item.get("adjustment_type", "")).lower()
            == "forward_split"
        ]
        stock_dividend = [
            item
            for item in records
            if str(item.get("adjustment_type", "")).lower()
            == "stock_dividend"
        ]

        latest_reverse = reverse[-1] if reverse else None
        consolidations = [
            value
            for value in (_consolidation(item) for item in reverse)
            if np.isfinite(value)
        ]

        index = result.index[pos]
        result.at[index, "split_history_query_success"] = 1.0
        result.at[index, "split_history_strict_prior"] = float(strict_prior)
        result.at[index, "split_any_730d"] = float(bool(records))
        result.at[index, "reverse_split_count_730d"] = float(len(reverse))
        result.at[index, "reverse_split_count_365d"] = float(
            len(reverse_365)
        )
        result.at[index, "forward_split_count_730d"] = float(len(forward))
        result.at[index, "stock_dividend_count_730d"] = float(
            len(stock_dividend)
        )
        if latest_reverse is not None:
            result.at[
                index, "split_days_since_latest_reverse"
            ] = float(
                (trading_day - latest_reverse["_execution_date"]).days
            )
            latest_consolidation = _consolidation(latest_reverse)
            if np.isfinite(latest_consolidation):
                result.at[
                    index, "split_latest_reverse_consolidation"
                ] = latest_consolidation
        if consolidations:
            result.at[
                index, "split_max_reverse_consolidation"
            ] = max(consolidations)

    return result


def coverage_row(frame: pd.DataFrame, month: str) -> dict[str, object]:
    feasible = frame.loc[execution_feasible_mask(frame)].copy()
    if feasible.empty:
        return {
            "month": month,
            "feasible_anchors": 0,
            "query_success": np.nan,
            "strict_prior": False,
            "reverse_split_anchor_rate_730d": np.nan,
            "reverse_split_anchors_730d": 0,
        }

    reverse = pd.to_numeric(
        feasible["reverse_split_count_730d"], errors="coerce"
    ).gt(0)
    return {
        "month": month,
        "feasible_anchors": int(len(feasible)),
        "query_success": float(
            pd.to_numeric(
                feasible["split_history_query_success"],
                errors="coerce",
            ).eq(1.0).mean()
        ),
        "strict_prior": bool(
            pd.to_numeric(
                feasible["split_history_strict_prior"],
                errors="coerce",
            ).eq(1.0).all()
        ),
        "reverse_split_anchor_rate_730d": float(reverse.mean()),
        "reverse_split_anchors_730d": int(reverse.sum()),
    }


def render_report(
    coverage: pd.DataFrame,
    client: MassiveClient,
    *,
    query_start: str,
    query_end: str,
    market_records: int,
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker Expanded-History Split Enrichment v2.8 ===",
            "source=Massive market-wide /stocks/v1/splits",
            f"global_query={query_start}..{query_end}",
            f"market_records={market_records}",
            "anchor_semantics=ticker match and D-730 <= execution_date < D",
            "no return-conditioned filtering",
            f"Massive_client_stats={client.stats.to_dict()}",
            "",
            coverage.to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.expanded_split_history_enrichment"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/massive-rest-v28-splits"),
    )
    args = parser.parse_args()

    frames: dict[str, pd.DataFrame] = {
        month: pd.read_parquet(path)
        for month, path in args.dataset
    }
    all_days = [
        pd.Timestamp(str(day)).date()
        for frame in frames.values()
        for day in frame.loc[
            execution_feasible_mask(frame), "trading_day"
        ].astype(str).unique()
    ]
    if not all_days:
        raise ValueError("no feasible anchor dates for split enrichment")

    query_start = min(all_days) - timedelta(days=LOOKBACK_DAYS)
    query_end = max(all_days) - timedelta(days=1)

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )
    records = client.splits_market(
        execution_date_gte=query_start,
        execution_date_lte=query_end,
    )
    events = _prepare_events(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict[str, object]] = []
    for month, frame in frames.items():
        enriched = enrich_frame(frame, events)
        enriched.to_parquet(
            args.output_dir / f"split-anchors-{month}.parquet",
            index=False,
            compression="zstd",
        )
        coverage_rows.append(coverage_row(enriched, month))

    coverage = pd.DataFrame(coverage_rows)
    report = render_report(
        coverage,
        client,
        query_start=query_start.isoformat(),
        query_end=query_end.isoformat(),
        market_records=len(records),
    )
    print(report, flush=True)

    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
