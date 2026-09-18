from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .massive_client import MassiveClient
from .short_interest_position_probe import PUBLICATION_SCHEDULE


SHORT_INTEREST_FEATURES = (
    "short_interest_latest",
    "short_interest_avg_daily_volume_latest",
    "short_interest_days_to_cover_latest",
    "short_interest_change_pct",
    "short_interest_avg_daily_volume_change_pct",
    "short_interest_days_to_cover_change",
    "short_interest_publication_age_days",
    "short_interest_reports_available",
)


def _number(value: object) -> float:
    item = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(item) if pd.notna(item) else np.nan


def _pct_change(latest: float, previous: float) -> float:
    if not np.isfinite(latest) or not np.isfinite(previous) or previous <= 0:
        return np.nan
    return float((latest / previous - 1.0) * 100.0)


def fetch_short_interest_history(
    frames: dict[str, pd.DataFrame],
    client: MassiveClient,
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    target_tickers: set[str] = set()
    for frame in frames.values():
        target_tickers.update(
            frame["ticker"].astype(str).str.upper().unique().tolist()
        )
    if not target_tickers:
        return {}, set()

    start = min(PUBLICATION_SCHEDULE)
    end = max(PUBLICATION_SCHEDULE)
    rows = client.short_interest_market(
        settlement_date_gte=start,
        settlement_date_lte=end,
    )

    history: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker not in target_tickers:
            continue
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
        history[ticker].append(item)

    for ticker in list(history):
        history[ticker].sort(
            key=lambda row: (
                row["_publication_day"],
                row["_settlement_day"],
            )
        )

    # A complete market-wide request covers zero-record tickers as valid misses.
    return dict(history), set(target_tickers)


def _feature_row(
    records: list[dict[str, object]],
    event_day: date,
) -> dict[str, float]:
    eligible = [
        row for row in records if row["_publication_day"] < event_day
    ]
    if any(row["_publication_day"] >= event_day for row in eligible):
        raise ValueError("short-interest publication-time violation")

    if not eligible:
        return {feature: np.nan for feature in SHORT_INTEREST_FEATURES}

    latest = eligible[-1]
    previous = eligible[-2] if len(eligible) >= 2 else None

    latest_short = _number(latest.get("short_interest"))
    latest_adv = _number(latest.get("avg_daily_volume"))
    latest_dtc = _number(latest.get("days_to_cover"))

    prev_short = (
        _number(previous.get("short_interest")) if previous else np.nan
    )
    prev_adv = (
        _number(previous.get("avg_daily_volume")) if previous else np.nan
    )
    prev_dtc = (
        _number(previous.get("days_to_cover")) if previous else np.nan
    )

    return {
        "short_interest_latest": latest_short,
        "short_interest_avg_daily_volume_latest": latest_adv,
        "short_interest_days_to_cover_latest": latest_dtc,
        "short_interest_change_pct": _pct_change(latest_short, prev_short),
        "short_interest_avg_daily_volume_change_pct": _pct_change(
            latest_adv, prev_adv
        ),
        "short_interest_days_to_cover_change": (
            float(latest_dtc - prev_dtc)
            if np.isfinite(latest_dtc) and np.isfinite(prev_dtc)
            else np.nan
        ),
        "short_interest_publication_age_days": float(
            (event_day - latest["_publication_day"]).days
        ),
        "short_interest_reports_available": float(len(eligible)),
    }


def enrich_frame(
    frame: pd.DataFrame,
    history: dict[str, list[dict[str, object]]],
    queried: set[str],
) -> pd.DataFrame:
    ticker_days = frame[["ticker", "trading_day"]].drop_duplicates().copy()
    rows: list[dict[str, object]] = []

    for item in ticker_days.itertuples(index=False):
        ticker = str(item.ticker).upper()
        trading_day = str(item.trading_day)
        event_day = pd.Timestamp(trading_day).date()
        if ticker not in queried:
            raise ValueError(f"short-interest query missing for ticker: {ticker}")
        rows.append(
            {
                "ticker": ticker,
                "trading_day": trading_day,
                "short_interest_query_complete": 1.0,
                **_feature_row(history.get(ticker, []), event_day),
            }
        )

    features = pd.DataFrame(rows)
    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result["trading_day"] = result["trading_day"].astype(str)
    return result.merge(
        features,
        on=["ticker", "trading_day"],
        how="left",
        validate="many_to_one",
    )


def coverage_summary(
    enriched: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, frame in enriched.items():
        ticker_days = frame[
            [
                "ticker",
                "trading_day",
                "short_interest_query_complete",
                *SHORT_INTEREST_FEATURES,
            ]
        ].drop_duplicates(["ticker", "trading_day"])
        rows.append(
            {
                "month": month,
                "rows": int(len(frame)),
                "ticker_days": int(len(ticker_days)),
                "query_complete_coverage": float(
                    pd.to_numeric(
                        ticker_days["short_interest_query_complete"],
                        errors="coerce",
                    ).eq(1.0).mean()
                ),
                **{
                    f"{feature}_coverage": float(
                        pd.to_numeric(
                            ticker_days[feature], errors="coerce"
                        ).notna().mean()
                    )
                    for feature in SHORT_INTEREST_FEATURES
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
            "=== MoneyMaker State Short-Interest Enrichment v1.3 ===",
            f"tickers_with_history={len(history)}",
            f"Massive_client_stats={client.stats.to_dict()}",
            "point_in_time_rule=official publication_date < state trading_day",
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
        prog="python -m victory_trader.state_short_interest_enrichment"
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
    history, queried = fetch_short_interest_history(frames, client)
    enriched = {
        label: enrich_frame(frame, history, queried)
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
            compression="zstd",
        )
    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
