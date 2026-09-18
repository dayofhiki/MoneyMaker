from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .massive_client import MassiveClient


SHORT_VOLUME_FEATURES = (
    "short_ratio_latest_prior",
    "short_ratio_mean5_prior",
    "short_ratio_mean20_prior",
    "short_ratio_latest_minus_mean5",
    "short_ratio_mean5_minus_mean20",
    "short_volume_latest_vs_mean5",
    "finra_total_volume_latest_vs_mean5",
    "short_volume_latest_age_days",
)

LOOKBACK_CALENDAR_DAYS = 45


def _finite(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else np.nan


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _full_mean(values: list[float], length: int) -> float:
    if len(values) < length:
        return np.nan
    arr = np.asarray(values[-length:], dtype=float)
    if not np.isfinite(arr).all():
        return np.nan
    return float(arr.mean())


def _feature_row(
    records: list[dict[str, object]],
    event_day: date,
) -> dict[str, float]:
    if not records:
        return {feature: np.nan for feature in SHORT_VOLUME_FEATURES}

    record_days = [record["_record_day"] for record in records]
    cut = bisect_left(record_days, event_day)
    prior = records[:cut]
    if not prior:
        return {feature: np.nan for feature in SHORT_VOLUME_FEATURES}

    latest = prior[-1]
    latest_day = latest["_record_day"]
    if latest_day >= event_day:
        raise ValueError(
            f"short-volume point-in-time violation: {latest_day} >= {event_day}"
        )

    ratios = [_finite(row.get("short_volume_ratio")) for row in prior]
    short_volumes = [_finite(row.get("short_volume")) for row in prior]
    total_volumes = [_finite(row.get("total_volume")) for row in prior]

    latest_ratio = ratios[-1]
    mean5_ratio = _full_mean(ratios, 5)
    mean20_ratio = _full_mean(ratios, 20)
    mean5_short = _full_mean(short_volumes, 5)
    mean5_total = _full_mean(total_volumes, 5)

    return {
        "short_ratio_latest_prior": latest_ratio,
        "short_ratio_mean5_prior": mean5_ratio,
        "short_ratio_mean20_prior": mean20_ratio,
        "short_ratio_latest_minus_mean5": (
            float(latest_ratio - mean5_ratio)
            if np.isfinite(latest_ratio) and np.isfinite(mean5_ratio)
            else np.nan
        ),
        "short_ratio_mean5_minus_mean20": (
            float(mean5_ratio - mean20_ratio)
            if np.isfinite(mean5_ratio) and np.isfinite(mean20_ratio)
            else np.nan
        ),
        "short_volume_latest_vs_mean5": _ratio(
            short_volumes[-1], mean5_short
        ),
        "finra_total_volume_latest_vs_mean5": _ratio(
            total_volumes[-1], mean5_total
        ),
        "short_volume_latest_age_days": float(
            (event_day - latest_day).days
        ),
    }


def _weekday_range(start: date, end: date):
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def fetch_short_volume_history(
    frames: dict[str, pd.DataFrame],
    client: MassiveClient,
) -> dict[str, list[dict[str, object]]]:
    if not frames:
        return {}

    target_tickers: set[str] = set()
    trading_days: list[date] = []
    for frame in frames.values():
        target_tickers.update(
            frame["ticker"].astype(str).str.upper().unique().tolist()
        )
        trading_days.extend(
            pd.to_datetime(frame["trading_day"].astype(str))
            .dt.date.unique()
            .tolist()
        )

    min_day = min(trading_days)
    max_day = max(trading_days)
    start = min_day - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    end = max_day - timedelta(days=1)

    rows = client.short_volume_market(
        date_gte=start,
        date_lte=end,
    )

    history: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker not in target_tickers:
            continue
        raw_date = row.get("date")
        if not raw_date:
            continue
        record_day = pd.Timestamp(str(raw_date)).date()
        if record_day < start or record_day > end:
            raise ValueError(
                "short-volume range violation: "
                f"{record_day} not in [{start}, {end}]"
            )
        item = dict(row)
        item["_record_day"] = record_day
        history[ticker].append(item)

    for ticker in list(history):
        history[ticker].sort(key=lambda row: row["_record_day"])
    return dict(history)


def enrich_frame(
    frame: pd.DataFrame,
    history: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    ticker_day = (
        frame[["ticker", "trading_day"]]
        .drop_duplicates()
        .copy()
    )
    feature_rows: list[dict[str, object]] = []
    for row in ticker_day.itertuples(index=False):
        ticker = str(row.ticker).upper()
        day_text = str(row.trading_day)
        event_day = pd.Timestamp(day_text).date()
        features = _feature_row(history.get(ticker, []), event_day)
        feature_rows.append(
            {
                "ticker": ticker,
                "trading_day": day_text,
                **features,
            }
        )

    features = pd.DataFrame(feature_rows)
    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result["trading_day"] = result["trading_day"].astype(str)
    result = result.merge(
        features,
        on=["ticker", "trading_day"],
        how="left",
        validate="many_to_one",
    )
    return result


def coverage_summary(
    enriched: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, frame in enriched.items():
        rows.append(
            {
                "month": month,
                "rows": int(len(frame)),
                "ticker_days": int(
                    frame[["ticker", "trading_day"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                **{
                    f"{feature}_coverage": float(
                        pd.to_numeric(frame[feature], errors="coerce")
                        .notna()
                        .mean()
                    )
                    for feature in SHORT_VOLUME_FEATURES
                },
            }
        )
    return pd.DataFrame(rows)


def render_report(
    coverage: pd.DataFrame,
    client: MassiveClient,
    history: dict[str, list[dict[str, object]]],
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State Short-Volume Enrichment v0.9 ===",
            f"tickers_with_history={len(history)}",
            f"Massive_client_stats={client.stats.to_dict()}",
            "point_in_time_rule=only record_date < state trading_day",
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
        prog="python -m victory_trader.state_short_volume_enrichment"
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
        default=Path("data/cache/massive-rest"),
    )
    args = parser.parse_args()

    frames = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )
    history = fetch_short_volume_history(frames, client)
    enriched = {
        label: enrich_frame(frame, history)
        for label, frame in frames.items()
    }
    coverage = coverage_summary(enriched)
    report = render_report(coverage, client, history)
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, frame in enriched.items():
        frame.to_parquet(
            args.output_dir / f"state-{label}.parquet",
            index=False,
        )
    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
