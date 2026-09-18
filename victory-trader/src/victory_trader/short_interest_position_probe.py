from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import load_event_dataset
from .config import load_settings
from .massive_client import MassiveClient
from .short_volume_supply_probe import _sample_signals


PUBLICATION_SCHEDULE: dict[date, date] = {
    date(2025, 11, 14): date(2025, 11, 25),
    date(2025, 11, 28): date(2025, 12, 9),
    date(2025, 12, 15): date(2025, 12, 24),
    date(2025, 12, 31): date(2026, 1, 12),
    date(2026, 1, 15): date(2026, 1, 27),
    date(2026, 1, 30): date(2026, 2, 10),
    date(2026, 2, 13): date(2026, 2, 25),
    date(2026, 2, 27): date(2026, 3, 10),
    date(2026, 3, 13): date(2026, 3, 24),
    date(2026, 3, 31): date(2026, 4, 10),
}

OVERALL_ANY_MIN = 0.80
MONTH_ANY_MIN = 0.70
OVERALL_TWO_MIN = 0.60


def _event_day(value: object) -> date:
    return pd.Timestamp(str(value)).date()


def _number(value: object) -> float:
    item = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(item) if pd.notna(item) else np.nan


def _prepare_ticker_history(rows: list[dict]) -> list[dict]:
    prepared: list[dict] = []
    for row in rows:
        raw = row.get("settlement_date")
        if not raw:
            continue
        settlement = pd.Timestamp(str(raw)).date()
        publication = PUBLICATION_SCHEDULE.get(settlement)
        if publication is None:
            continue
        item = dict(row)
        item["_settlement_day"] = settlement
        item["_publication_day"] = publication
        prepared.append(item)
    return sorted(
        prepared,
        key=lambda row: (row["_publication_day"], row["_settlement_day"]),
    )


def _eligible_history(
    records: list[dict],
    event_day: date,
) -> list[dict]:
    eligible = [
        row for row in records if row["_publication_day"] < event_day
    ]
    if any(row["_publication_day"] >= event_day for row in eligible):
        raise ValueError("short-interest publication-time violation")
    return eligible


def _event_metrics(
    *,
    month: str,
    trading_day: str,
    ticker: str,
    threshold_pct: float,
    records: list[dict],
) -> dict[str, object]:
    event_day = _event_day(trading_day)
    eligible = _eligible_history(records, event_day)
    latest = eligible[-1] if eligible else None
    previous = eligible[-2] if len(eligible) >= 2 else None

    latest_si = _number(latest.get("short_interest")) if latest else np.nan
    previous_si = (
        _number(previous.get("short_interest")) if previous else np.nan
    )
    change_pct = (
        float((latest_si / previous_si - 1.0) * 100.0)
        if np.isfinite(latest_si)
        and np.isfinite(previous_si)
        and previous_si > 0
        else np.nan
    )

    return {
        "month": month,
        "trading_day": trading_day,
        "ticker": ticker,
        "threshold_pct": threshold_pct,
        "eligible_records": int(len(eligible)),
        "has_published_record": bool(latest),
        "has_two_published_records": bool(len(eligible) >= 2),
        "latest_settlement_date": (
            latest["_settlement_day"].isoformat() if latest else None
        ),
        "latest_publication_date": (
            latest["_publication_day"].isoformat() if latest else None
        ),
        "latest_publication_age_days": (
            float((event_day - latest["_publication_day"]).days)
            if latest
            else np.nan
        ),
        "latest_short_interest": latest_si,
        "latest_avg_daily_volume": (
            _number(latest.get("avg_daily_volume"))
            if latest
            else np.nan
        ),
        "latest_days_to_cover": (
            _number(latest.get("days_to_cover"))
            if latest
            else np.nan
        ),
        "previous_short_interest": previous_si,
        "latest_short_interest_change_pct": change_pct,
    }


def run_probe(
    monthly_frames: dict[str, pd.DataFrame],
    client: MassiveClient,
) -> pd.DataFrame:
    sample = _sample_signals(monthly_frames)
    if sample.empty:
        return pd.DataFrame()

    histories: dict[str, list[dict]] = {}
    start = min(PUBLICATION_SCHEDULE)
    end = max(PUBLICATION_SCHEDULE)
    for ticker in sorted(sample["ticker"].astype(str).str.upper().unique()):
        histories[ticker] = _prepare_ticker_history(
            client.short_interest(
                ticker,
                settlement_date_gte=start,
                settlement_date_lte=end,
            )
        )

    rows: list[dict[str, object]] = []
    for _, event in sample.iterrows():
        ticker = str(event["ticker"]).upper()
        rows.append(
            _event_metrics(
                month=str(event["month"]),
                trading_day=str(event["trading_day"]),
                ticker=ticker,
                threshold_pct=float(event["threshold_pct"]),
                records=histories.get(ticker, []),
            )
        )
    return pd.DataFrame(rows)


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    monthly = (
        frame.groupby("month")
        .agg(
            sampled=("ticker", "size"),
            any_published_coverage=("has_published_record", "mean"),
            two_published_coverage=("has_two_published_records", "mean"),
            median_publication_age_days=(
                "latest_publication_age_days",
                "median",
            ),
            days_to_cover_coverage=("latest_days_to_cover", lambda s: s.notna().mean()),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "month": "OVERALL",
                "sampled": int(len(frame)),
                "any_published_coverage": float(
                    frame["has_published_record"].mean()
                ),
                "two_published_coverage": float(
                    frame["has_two_published_records"].mean()
                ),
                "median_publication_age_days": float(
                    frame["latest_publication_age_days"].median()
                ),
                "days_to_cover_coverage": float(
                    frame["latest_days_to_cover"].notna().mean()
                ),
            }
        ]
    )
    return pd.concat([monthly, overall], ignore_index=True)


def success_check(frame: pd.DataFrame) -> dict[str, bool]:
    if frame.empty:
        return {
            "overall_any_published_at_least_80pct": False,
            "every_month_any_published_at_least_70pct": False,
            "overall_two_published_at_least_60pct": False,
            "all_pass": False,
        }

    monthly = frame.groupby("month")["has_published_record"].mean()
    checks = {
        "overall_any_published_at_least_80pct": bool(
            frame["has_published_record"].mean() >= OVERALL_ANY_MIN
        ),
        "every_month_any_published_at_least_70pct": bool(
            len(monthly) == 3 and monthly.ge(MONTH_ANY_MIN).all()
        ),
        "overall_two_published_at_least_60pct": bool(
            frame["has_two_published_records"].mean() >= OVERALL_TWO_MIN
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(frame: pd.DataFrame, client: MassiveClient) -> str:
    if frame.empty:
        return (
            "=== MoneyMaker Historical Short-Interest Position Probe v1.2 ===\n"
            "No sampled candidate signals."
        )

    summary = coverage_summary(frame)
    checks = success_check(frame)
    return "\n".join(
        [
            "=== MoneyMaker Historical Short-Interest Position Probe v1.2 ===",
            f"sampled_events={len(frame)}",
            f"unique_tickers={frame['ticker'].nunique()}",
            "point_in_time_rule=official FINRA publication date strictly before event day",
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
            "=== Sampled events ===",
            frame.to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.short_interest_position_probe"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
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
    result = run_probe(frames, client)
    report = render_report(result, client)
    print(report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    result.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
