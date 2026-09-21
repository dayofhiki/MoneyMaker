from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .massive_client import MassiveClient


SOURCE_POLICY = "feasible_earliest_15m_cap1"
SAMPLE_PER_MONTH = 25
LOOKBACK_DAYS = 730
REQUEST_SUCCESS_MIN = 0.95
MIN_REVERSE_TOTAL = 10
MIN_REVERSE_PER_MONTH = 2


def deterministic_sample(attempts: pd.DataFrame) -> pd.DataFrame:
    frame = attempts.loc[attempts["policy"].eq(SOURCE_POLICY)].copy()
    frame["month"] = frame["month"].astype(str)
    frame["trading_day"] = frame["trading_day"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    time_col = "decision_t" if "decision_t" in frame.columns else "t"

    frame = (
        frame.sort_values(
            ["month", "trading_day", "ticker", time_col],
            kind="stable",
        )
        .drop_duplicates(["month", "trading_day", "ticker"])
        .reset_index(drop=True)
    )

    samples: list[pd.DataFrame] = []
    for month, group in frame.groupby("month", sort=True):
        ordered = group.sort_values(
            ["trading_day", "ticker", time_col],
            kind="stable",
        ).reset_index(drop=True)
        if len(ordered) < SAMPLE_PER_MONTH:
            raise ValueError(
                f"{month} has only {len(ordered)} unique ticker-day attempts"
            )
        indexes = np.linspace(
            0,
            len(ordered) - 1,
            SAMPLE_PER_MONTH,
            dtype=int,
        )
        samples.append(ordered.iloc[indexes].copy())

    if len(samples) != 3:
        raise ValueError(f"expected 3 development months, found {len(samples)}")
    return pd.concat(samples, ignore_index=True)


def _positive(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or float(numeric) <= 0:
        return np.nan
    return float(numeric)


def _ratio(split_from: object, split_to: object) -> tuple[float, float]:
    before = _positive(split_from)
    after = _positive(split_to)
    if not np.isfinite(before) or not np.isfinite(after):
        return np.nan, np.nan
    return after / before, before / after


def probe_rows(sample: pd.DataFrame, client: MassiveClient) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for item in sample.itertuples(index=False):
        trading_day = pd.Timestamp(str(item.trading_day)).date()
        start = trading_day - timedelta(days=LOOKBACK_DAYS)
        end = trading_day - timedelta(days=1)
        ticker = str(item.ticker).upper()

        request_success = True
        error_type = ""
        records: list[dict[str, object]] = []
        try:
            records = client.splits(
                ticker,
                execution_date_gte=start,
                execution_date_lte=end,
            )
        except Exception as exc:
            request_success = False
            error_type = type(exc).__name__

        valid_records: list[dict[str, object]] = []
        strict_prior = True
        for record in records:
            raw_date = record.get("execution_date")
            try:
                execution_date = pd.Timestamp(str(raw_date)).date()
            except Exception:
                strict_prior = False
                continue
            if execution_date >= trading_day:
                strict_prior = False
            valid_records.append(
                {
                    **record,
                    "_execution_date": execution_date,
                }
            )

        reverse = [
            record
            for record in valid_records
            if str(record.get("adjustment_type", "")).lower()
            == "reverse_split"
        ]
        forward = [
            record
            for record in valid_records
            if str(record.get("adjustment_type", "")).lower()
            == "forward_split"
        ]
        stock_dividend = [
            record
            for record in valid_records
            if str(record.get("adjustment_type", "")).lower()
            == "stock_dividend"
        ]

        reverse_365 = [
            record
            for record in reverse
            if (trading_day - record["_execution_date"]).days <= 365
        ]

        latest_split = max(
            (record["_execution_date"] for record in valid_records),
            default=None,
        )
        latest_reverse = max(
            (record["_execution_date"] for record in reverse),
            default=None,
        )

        reverse_ratios = [
            _ratio(record.get("split_from"), record.get("split_to"))
            for record in reverse
        ]
        latest_reverse_record = (
            max(reverse, key=lambda record: record["_execution_date"])
            if reverse
            else None
        )
        if latest_reverse_record is not None:
            latest_ratio, latest_consolidation = _ratio(
                latest_reverse_record.get("split_from"),
                latest_reverse_record.get("split_to"),
            )
        else:
            latest_ratio, latest_consolidation = np.nan, np.nan

        consolidations = [
            consolidation
            for _, consolidation in reverse_ratios
            if np.isfinite(consolidation)
        ]

        rows.append(
            {
                "month": str(item.month),
                "trading_day": trading_day.isoformat(),
                "ticker": ticker,
                "query_start": start.isoformat(),
                "query_end": end.isoformat(),
                "request_success": float(request_success),
                "error_type": error_type,
                "strict_prior_records": float(strict_prior),
                "split_count_730d": len(valid_records),
                "reverse_split_count_730d": len(reverse),
                "reverse_split_count_365d": len(reverse_365),
                "forward_split_count_730d": len(forward),
                "stock_dividend_count_730d": len(stock_dividend),
                "days_since_latest_split": (
                    (trading_day - latest_split).days
                    if latest_split is not None
                    else np.nan
                ),
                "days_since_latest_reverse_split": (
                    (trading_day - latest_reverse).days
                    if latest_reverse is not None
                    else np.nan
                ),
                "latest_reverse_ratio": latest_ratio,
                "latest_reverse_consolidation_factor": latest_consolidation,
                "max_reverse_consolidation_factor": (
                    max(consolidations) if consolidations else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def coverage_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for month, group in rows.groupby("month", sort=True):
        reverse_any = pd.to_numeric(
            group["reverse_split_count_730d"], errors="coerce"
        ).gt(0)
        summaries.append(
            {
                "month": month,
                "sampled": int(len(group)),
                "request_success": float(
                    pd.to_numeric(
                        group["request_success"], errors="coerce"
                    ).eq(1.0).mean()
                ),
                "strict_prior_rate": float(
                    pd.to_numeric(
                        group["strict_prior_records"], errors="coerce"
                    ).eq(1.0).mean()
                ),
                "any_split_rate_730d": float(
                    pd.to_numeric(
                        group["split_count_730d"], errors="coerce"
                    ).gt(0).mean()
                ),
                "reverse_split_rate_730d": float(reverse_any.mean()),
                "reverse_split_anchors_730d": int(reverse_any.sum()),
            }
        )
    return pd.DataFrame(summaries)


def success_check(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, bool]:
    reverse_total = int(
        pd.to_numeric(
            rows["reverse_split_count_730d"], errors="coerce"
        ).gt(0).sum()
    )
    checks = {
        "request_success_every_month_ge_95pct": bool(
            len(summary) == 3
            and summary["request_success"].ge(REQUEST_SUCCESS_MIN).all()
        ),
        "all_persisted_dates_strictly_prior": bool(
            pd.to_numeric(
                rows["strict_prior_records"], errors="coerce"
            ).eq(1.0).all()
        ),
        "reverse_split_support_total_ge_10": reverse_total
        >= MIN_REVERSE_TOTAL,
        "reverse_split_support_each_month_ge_2": bool(
            len(summary) == 3
            and summary["reverse_split_anchors_730d"]
            .ge(MIN_REVERSE_PER_MONTH)
            .all()
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    checks: dict[str, bool],
    client: MassiveClient,
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker Point-in-Time Split-History Probe v0.1 ===",
            f"sampled_rows={len(rows)}",
            (
                "source=/stocks/v1/splits; ticker-specific; "
                "execution_date D-730 through D-1 only"
            ),
            f"Massive_client_stats={client.stats.to_dict()}",
            "",
            "=== Coverage and support ===",
            summary.to_string(index=False),
            "",
            "=== Pre-registered go/no-go ===",
            pd.DataFrame(
                [
                    {"criterion": key, "pass": value}
                    for key, value in checks.items()
                ]
            ).to_string(index=False),
            "",
            "=== Probe rows ===",
            rows.to_string(index=False),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.point_in_time_split_history_probe"
    )
    parser.add_argument("--attempts-csv", type=Path, required=True)
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/massive-rest"),
    )
    args = parser.parse_args()

    attempts = pd.read_csv(args.attempts_csv)
    sample = deterministic_sample(attempts)

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )
    rows = probe_rows(sample, client)
    summary = coverage_summary(rows)
    checks = success_check(rows, summary)
    report = render_report(rows, summary, checks, client)
    print(report, flush=True)

    for path in (args.rows_csv, args.coverage_csv, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.rows_csv, index=False)
    summary.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
