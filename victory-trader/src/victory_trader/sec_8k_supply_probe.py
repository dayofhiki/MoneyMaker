from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import load_event_dataset
from .config import load_settings
from .massive_client import MassiveClient
from .short_volume_supply_probe import _sample_signals


LOOKBACK_CALENDAR_DAYS = 30
OVERALL_30D_MIN = 0.20
MONTH_30D_MIN = 0.10
CATEGORY_COMPLETENESS_MIN = 0.80


def _event_day(value: object) -> date:
    return pd.Timestamp(str(value)).date()


def _safe_prior_disclosures(
    client: MassiveClient,
    ticker: str,
    event_day: date,
) -> list[dict]:
    rows = client.eight_k_disclosures(
        ticker,
        filing_date_gte=event_day - timedelta(days=LOOKBACK_CALENDAR_DAYS),
        filing_date_lte=event_day - timedelta(days=1),
    )
    cleaned: list[dict] = []
    for row in rows:
        raw = row.get("filing_date")
        if not raw:
            continue
        filing_day = pd.Timestamp(str(raw)).date()
        if filing_day >= event_day:
            raise ValueError(
                "8-K point-in-time violation: "
                f"{ticker} event={event_day} filing={filing_day}"
            )
        item = dict(row)
        item["_filing_day"] = filing_day
        cleaned.append(item)
    return sorted(cleaned, key=lambda row: row["_filing_day"])


def _nonempty(value: object) -> bool:
    return bool(str(value or "").strip())


def _event_metrics(
    *,
    month: str,
    trading_day: str,
    ticker: str,
    threshold_pct: float,
    disclosures: list[dict],
) -> dict[str, object]:
    event_day = _event_day(trading_day)
    accessions = {
        str(row.get("accession_number") or "")
        for row in disclosures
        if str(row.get("accession_number") or "")
    }
    filing_days = [row["_filing_day"] for row in disclosures]
    latest_day = max(filing_days) if filing_days else None
    latest_age = (event_day - latest_day).days if latest_day else np.nan

    primary = {
        str(row.get("primary_category"))
        for row in disclosures
        if _nonempty(row.get("primary_category"))
    }
    secondary = {
        str(row.get("secondary_category"))
        for row in disclosures
        if _nonempty(row.get("secondary_category"))
    }
    tertiary = {
        str(row.get("tertiary_category"))
        for row in disclosures
        if _nonempty(row.get("tertiary_category"))
    }

    return {
        "month": month,
        "trading_day": trading_day,
        "ticker": ticker,
        "threshold_pct": threshold_pct,
        "unique_filings_30d": int(len(accessions)),
        "disclosures_30d": int(len(disclosures)),
        "has_filing_within_7d": bool(
            latest_day and latest_age <= 7
        ),
        "has_filing_within_30d": bool(latest_day),
        "latest_filing_date": latest_day.isoformat() if latest_day else None,
        "latest_filing_age_days": float(latest_age)
        if np.isfinite(latest_age)
        else np.nan,
        "distinct_primary_categories": int(len(primary)),
        "distinct_secondary_categories": int(len(secondary)),
        "distinct_tertiary_categories": int(len(tertiary)),
    }


def run_probe(
    monthly_frames: dict[str, pd.DataFrame],
    client: MassiveClient,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = _sample_signals(monthly_frames)
    event_rows: list[dict[str, object]] = []
    disclosure_rows: list[dict[str, object]] = []

    for _, event in sample.iterrows():
        ticker = str(event["ticker"]).upper()
        event_day = _event_day(event["trading_day"])
        disclosures = _safe_prior_disclosures(client, ticker, event_day)
        event_rows.append(
            _event_metrics(
                month=str(event["month"]),
                trading_day=str(event["trading_day"]),
                ticker=ticker,
                threshold_pct=float(event["threshold_pct"]),
                disclosures=disclosures,
            )
        )
        for row in disclosures:
            disclosure_rows.append(
                {
                    "month": str(event["month"]),
                    "trading_day": str(event["trading_day"]),
                    "ticker": ticker,
                    "filing_date": row["_filing_day"].isoformat(),
                    "accession_number": row.get("accession_number"),
                    "primary_category": row.get("primary_category"),
                    "secondary_category": row.get("secondary_category"),
                    "tertiary_category": row.get("tertiary_category"),
                    "category_complete": bool(
                        _nonempty(row.get("primary_category"))
                        and _nonempty(row.get("secondary_category"))
                        and _nonempty(row.get("tertiary_category"))
                    ),
                }
            )

    return pd.DataFrame(event_rows), pd.DataFrame(disclosure_rows)


def coverage_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    monthly = (
        events.groupby("month")
        .agg(
            sampled=("ticker", "size"),
            filing_7d_coverage=("has_filing_within_7d", "mean"),
            filing_30d_coverage=("has_filing_within_30d", "mean"),
            median_unique_filings_30d=("unique_filings_30d", "median"),
            median_disclosures_30d=("disclosures_30d", "median"),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "month": "OVERALL",
                "sampled": int(len(events)),
                "filing_7d_coverage": float(
                    events["has_filing_within_7d"].mean()
                ),
                "filing_30d_coverage": float(
                    events["has_filing_within_30d"].mean()
                ),
                "median_unique_filings_30d": float(
                    events["unique_filings_30d"].median()
                ),
                "median_disclosures_30d": float(
                    events["disclosures_30d"].median()
                ),
            }
        ]
    )
    return pd.concat([monthly, overall], ignore_index=True)


def success_check(
    events: pd.DataFrame,
    disclosures: pd.DataFrame,
) -> dict[str, bool]:
    if events.empty:
        return {
            "overall_30d_coverage_at_least_20pct": False,
            "every_month_30d_coverage_at_least_10pct": False,
            "category_completeness_at_least_80pct": False,
            "all_pass": False,
        }
    monthly = events.groupby("month")["has_filing_within_30d"].mean()
    completeness = (
        float(disclosures["category_complete"].mean())
        if not disclosures.empty
        else 0.0
    )
    checks = {
        "overall_30d_coverage_at_least_20pct": bool(
            float(events["has_filing_within_30d"].mean())
            >= OVERALL_30D_MIN
        ),
        "every_month_30d_coverage_at_least_10pct": bool(
            len(monthly) == 3
            and monthly.ge(MONTH_30D_MIN).all()
        ),
        "category_completeness_at_least_80pct": bool(
            completeness >= CATEGORY_COMPLETENESS_MIN
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def _category_table(
    disclosures: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    if disclosures.empty:
        return pd.DataFrame(columns=[column, "count"])
    values = disclosures.loc[
        disclosures[column].astype(str).str.len().gt(0),
        column,
    ]
    return (
        values.value_counts()
        .rename_axis(column)
        .reset_index(name="count")
        .head(20)
    )


def render_report(
    events: pd.DataFrame,
    disclosures: pd.DataFrame,
    client: MassiveClient,
) -> str:
    if events.empty:
        return "=== MoneyMaker Historical 8-K Disclosure Supply Probe v0.1 ===\nNo sampled candidate signals."

    summary = coverage_summary(events)
    checks = success_check(events, disclosures)
    completeness = (
        float(disclosures["category_complete"].mean())
        if not disclosures.empty
        else 0.0
    )
    return "\n".join(
        [
            "=== MoneyMaker Historical 8-K Disclosure Supply Probe v0.1 ===",
            f"sampled_events={len(events)}",
            f"unique_tickers={events['ticker'].nunique()}",
            f"retrieved_disclosure_rows={len(disclosures)}",
            f"category_completeness={completeness:.4f}",
            "point_in_time_rule=filing_date strictly before candidate trading day",
            f"Massive_client_stats={client.stats.to_dict()}",
            "",
            "=== Coverage ===",
            summary.to_string(index=False),
            "",
            "=== Pre-registered coverage check ===",
            pd.DataFrame(
                [
                    {"criterion": key, "pass": value}
                    for key, value in checks.items()
                ]
            ).to_string(index=False),
            "",
            "=== Top primary categories ===",
            _category_table(disclosures, "primary_category").to_string(index=False),
            "",
            "=== Top secondary categories ===",
            _category_table(disclosures, "secondary_category").to_string(index=False),
            "",
            "=== Top tertiary categories ===",
            _category_table(disclosures, "tertiary_category").to_string(index=False),
            "",
            "=== Sampled events ===",
            events.to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.sec_8k_supply_probe"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--disclosures-csv", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/massive-rest"),
    )
    args = parser.parse_args()

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )
    frames = {
        label: load_event_dataset(path)
        for label, path in args.dataset
    }
    events, disclosures = run_probe(frames, client)
    report = render_report(events, disclosures, client)
    print(report)

    for path in (args.report, args.events_csv, args.disclosures_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    events.to_csv(args.events_csv, index=False)
    disclosures.to_csv(args.disclosures_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
