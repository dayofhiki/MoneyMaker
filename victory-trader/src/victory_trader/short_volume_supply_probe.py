from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import load_event_dataset
from .candidate_v2_entry_research import SIGNAL_CONFIG, _score_signal_only
from .candidate_v2_research import _fit_threshold_params
from .config import load_settings
from .massive_client import MassiveClient


SAMPLE_PER_MONTH = 25
LOOKBACK_CALENDAR_DAYS = 30
OVERALL_RECENT_7D_MIN = 0.90
MONTH_RECENT_7D_MIN = 0.80
FIVE_RECORD_COVERAGE_MIN = 0.80


def _sample_signals(monthly_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )
        params = _fit_threshold_params(train, SIGNAL_CONFIG)
        scored = _score_signal_only(holdout, params)
        selected = scored.loc[scored["candidate_v2_signal"].astype(bool)].copy()
        selected["month"] = holdout_month
        selected = selected.sort_values(
            ["trading_day", "threshold_pct", "ticker", "timestamp_ms"],
            kind="stable",
        ).reset_index(drop=True)
        if len(selected) > SAMPLE_PER_MONTH:
            positions = np.linspace(
                0,
                len(selected) - 1,
                SAMPLE_PER_MONTH,
                dtype=int,
            )
            selected = selected.iloc[positions].copy()
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _event_day(value: object) -> date:
    return pd.Timestamp(str(value)).date()


def _safe_prior_short_volume(
    client: MassiveClient,
    ticker: str,
    event_day: date,
) -> list[dict]:
    rows = client.short_volume(
        ticker,
        date_gte=event_day - timedelta(days=LOOKBACK_CALENDAR_DAYS),
        date_lte=event_day - timedelta(days=1),
    )
    cleaned: list[dict] = []
    for row in rows:
        raw_date = row.get("date")
        if not raw_date:
            continue
        record_day = pd.Timestamp(str(raw_date)).date()
        if record_day >= event_day:
            raise ValueError(
                "short-volume point-in-time violation: "
                f"{ticker} event={event_day} record={record_day}"
            )
        item = dict(row)
        item["_record_day"] = record_day
        cleaned.append(item)
    return sorted(cleaned, key=lambda row: row["_record_day"])


def _finite_number(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else np.nan


def _event_metrics(
    *,
    month: str,
    trading_day: str,
    ticker: str,
    threshold_pct: float,
    records: list[dict],
) -> dict[str, object]:
    event_day = _event_day(trading_day)
    latest = records[-1] if records else None
    latest_day = latest["_record_day"] if latest else None
    age_days = (event_day - latest_day).days if latest_day else np.nan

    ratios = np.asarray(
        [_finite_number(row.get("short_volume_ratio")) for row in records],
        dtype=float,
    )
    finite_ratios = ratios[np.isfinite(ratios)]
    latest_five = finite_ratios[-5:]
    five_mean = (
        float(np.mean(latest_five))
        if len(latest_five) == 5
        else np.nan
    )
    latest_ratio = (
        _finite_number(latest.get("short_volume_ratio"))
        if latest
        else np.nan
    )

    return {
        "month": month,
        "trading_day": trading_day,
        "ticker": ticker,
        "threshold_pct": threshold_pct,
        "records_30d": int(len(records)),
        "latest_record_date": latest_day.isoformat() if latest_day else None,
        "latest_record_age_days": float(age_days)
        if np.isfinite(age_days)
        else np.nan,
        "has_prior_within_1d": bool(latest_day and age_days <= 1),
        "has_prior_within_3d": bool(latest_day and age_days <= 3),
        "has_prior_within_7d": bool(latest_day and age_days <= 7),
        "has_five_records": bool(len(finite_ratios) >= 5),
        "latest_short_volume_ratio": latest_ratio,
        "latest5_short_volume_ratio_mean": five_mean,
        "latest_minus_latest5_mean": (
            float(latest_ratio - five_mean)
            if np.isfinite(latest_ratio) and np.isfinite(five_mean)
            else np.nan
        ),
        "latest_short_volume": (
            _finite_number(latest.get("short_volume"))
            if latest
            else np.nan
        ),
        "latest_total_volume": (
            _finite_number(latest.get("total_volume"))
            if latest
            else np.nan
        ),
    }


def run_probe(
    monthly_frames: dict[str, pd.DataFrame],
    client: MassiveClient,
) -> pd.DataFrame:
    sample = _sample_signals(monthly_frames)
    rows: list[dict[str, object]] = []

    for _, event in sample.iterrows():
        event_day = _event_day(event["trading_day"])
        ticker = str(event["ticker"]).upper()
        records = _safe_prior_short_volume(client, ticker, event_day)
        rows.append(
            _event_metrics(
                month=str(event["month"]),
                trading_day=str(event["trading_day"]),
                ticker=ticker,
                threshold_pct=float(event["threshold_pct"]),
                records=records,
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
            recent_1d_coverage=("has_prior_within_1d", "mean"),
            recent_3d_coverage=("has_prior_within_3d", "mean"),
            recent_7d_coverage=("has_prior_within_7d", "mean"),
            five_record_coverage=("has_five_records", "mean"),
            median_latest_age_days=("latest_record_age_days", "median"),
            median_records_30d=("records_30d", "median"),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "month": "OVERALL",
                "sampled": int(len(frame)),
                "recent_1d_coverage": float(
                    frame["has_prior_within_1d"].mean()
                ),
                "recent_3d_coverage": float(
                    frame["has_prior_within_3d"].mean()
                ),
                "recent_7d_coverage": float(
                    frame["has_prior_within_7d"].mean()
                ),
                "five_record_coverage": float(
                    frame["has_five_records"].mean()
                ),
                "median_latest_age_days": float(
                    frame["latest_record_age_days"].median()
                ),
                "median_records_30d": float(
                    frame["records_30d"].median()
                ),
            }
        ]
    )
    return pd.concat([monthly, overall], ignore_index=True)


def success_check(frame: pd.DataFrame) -> dict[str, bool]:
    if frame.empty:
        return {
            "overall_recent_7d_at_least_90pct": False,
            "every_month_recent_7d_at_least_80pct": False,
            "five_record_coverage_at_least_80pct": False,
            "all_pass": False,
        }

    monthly = frame.groupby("month")["has_prior_within_7d"].mean()
    checks = {
        "overall_recent_7d_at_least_90pct": bool(
            float(frame["has_prior_within_7d"].mean())
            >= OVERALL_RECENT_7D_MIN
        ),
        "every_month_recent_7d_at_least_80pct": bool(
            len(monthly) == 3
            and monthly.ge(MONTH_RECENT_7D_MIN).all()
        ),
        "five_record_coverage_at_least_80pct": bool(
            float(frame["has_five_records"].mean())
            >= FIVE_RECORD_COVERAGE_MIN
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    frame: pd.DataFrame,
    client: MassiveClient,
) -> str:
    if frame.empty:
        return (
            "=== MoneyMaker Historical Short-Volume Supply Probe v0.1 ===\n"
            "No sampled candidate signals."
        )

    summary = coverage_summary(frame)
    checks = success_check(frame)
    display_columns = [
        "month",
        "trading_day",
        "ticker",
        "threshold_pct",
        "records_30d",
        "latest_record_date",
        "latest_record_age_days",
        "has_prior_within_7d",
        "has_five_records",
        "latest_short_volume_ratio",
        "latest5_short_volume_ratio_mean",
        "latest_minus_latest5_mean",
    ]

    return "\n".join(
        [
            "=== MoneyMaker Historical Short-Volume Supply Probe v0.1 ===",
            f"sampled_events={len(frame)}",
            f"unique_tickers={frame['ticker'].nunique()}",
            "point_in_time_rule=records strictly before candidate trading day",
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
            frame[display_columns].to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.short_volume_supply_probe"
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
