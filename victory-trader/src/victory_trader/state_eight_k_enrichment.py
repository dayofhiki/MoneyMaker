from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .massive_client import MassiveClient


LOOKBACK_CALENDAR_DAYS = 30

PRIMARY_CATEGORIES = (
    "capital_and_financing",
    "leadership_and_governance",
    "shareholder_activity",
    "strategic_transactions",
    "financial_results",
    "operations_and_strategy",
    "risk_events",
    "regulatory_and_compliance",
)

EIGHT_K_FEATURES = (
    "eight_k_unique_filings_30d",
    "eight_k_disclosures_30d",
    "eight_k_has_filing_7d",
    "eight_k_latest_filing_age_days",
    "eight_k_distinct_primary_categories_30d",
    *tuple(
        f"eight_k_primary_{category}_count_30d"
        for category in PRIMARY_CATEGORIES
    ),
)


def _filing_day(row: dict[str, object]) -> date | None:
    raw = row.get("filing_date")
    if not raw:
        return None
    return pd.Timestamp(str(raw)).date()


def fetch_eight_k_history(
    frames: dict[str, pd.DataFrame],
    client: MassiveClient,
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
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

    if not trading_days:
        return {}, set()

    start = min(trading_days) - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    end = max(trading_days) - timedelta(days=1)
    history: dict[str, list[dict[str, object]]] = defaultdict(list)
    queried: set[str] = set()

    for ticker in sorted(target_tickers):
        rows = client.eight_k_disclosures(
            ticker,
            filing_date_gte=start,
            filing_date_lte=end,
        )
        queried.add(ticker)
        for row in rows:
            day = _filing_day(row)
            if day is None:
                continue
            if day < start or day > end:
                raise ValueError(
                    "8-K global request date violation: "
                    f"{ticker} {day} not in [{start}, {end}]"
                )
            item = dict(row)
            item["_filing_day"] = day
            history[ticker].append(item)

    for ticker in list(history):
        history[ticker].sort(key=lambda row: row["_filing_day"])
    return dict(history), queried


def _feature_row(
    records: list[dict[str, object]],
    event_day: date,
) -> dict[str, float]:
    start = event_day - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    prior = [
        row
        for row in records
        if start <= row["_filing_day"] < event_day
    ]
    if any(row["_filing_day"] >= event_day for row in prior):
        raise ValueError("8-K point-in-time violation")

    accessions = {
        str(row.get("accession_number") or "")
        for row in prior
        if str(row.get("accession_number") or "")
    }
    filing_days = [row["_filing_day"] for row in prior]
    latest = max(filing_days) if filing_days else None
    primary = {
        str(row.get("primary_category") or "")
        for row in prior
        if str(row.get("primary_category") or "")
    }

    result: dict[str, float] = {
        "eight_k_unique_filings_30d": float(len(accessions)),
        "eight_k_disclosures_30d": float(len(prior)),
        "eight_k_has_filing_7d": float(
            bool(latest and (event_day - latest).days <= 7)
        ),
        "eight_k_latest_filing_age_days": (
            float((event_day - latest).days)
            if latest is not None
            else np.nan
        ),
        "eight_k_distinct_primary_categories_30d": float(len(primary)),
    }

    for category in PRIMARY_CATEGORIES:
        result[f"eight_k_primary_{category}_count_30d"] = float(
            sum(
                str(row.get("primary_category") or "") == category
                for row in prior
            )
        )
    return result


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
            raise ValueError(f"8-K query missing for ticker: {ticker}")
        features = _feature_row(history.get(ticker, []), event_day)
        rows.append(
            {
                "ticker": ticker,
                "trading_day": trading_day,
                "eight_k_query_complete": 1.0,
                **features,
            }
        )

    feature_frame = pd.DataFrame(rows)
    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result["trading_day"] = result["trading_day"].astype(str)
    return result.merge(
        feature_frame,
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
                "eight_k_query_complete",
                "eight_k_unique_filings_30d",
                "eight_k_has_filing_7d",
            ]
        ].drop_duplicates(["ticker", "trading_day"])
        rows.append(
            {
                "month": month,
                "rows": int(len(frame)),
                "ticker_days": int(len(ticker_days)),
                "query_complete_coverage": float(
                    pd.to_numeric(
                        ticker_days["eight_k_query_complete"],
                        errors="coerce",
                    ).eq(1.0).mean()
                ),
                "filing_30d_prevalence": float(
                    pd.to_numeric(
                        ticker_days["eight_k_unique_filings_30d"],
                        errors="coerce",
                    ).gt(0).mean()
                ),
                "filing_7d_prevalence": float(
                    pd.to_numeric(
                        ticker_days["eight_k_has_filing_7d"],
                        errors="coerce",
                    ).eq(1.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def render_report(
    coverage: pd.DataFrame,
    client: MassiveClient,
    queried: set[str],
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State 8-K Enrichment v1.1 ===",
            f"queried_tickers={len(queried)}",
            f"Massive_client_stats={client.stats.to_dict()}",
            "point_in_time_rule=D-30 <= filing_date < D",
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
        prog="python -m victory_trader.state_eight_k_enrichment"
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
    history, queried = fetch_eight_k_history(frames, client)
    enriched = {
        label: enrich_frame(frame, history, queried)
        for label, frame in frames.items()
    }
    coverage = coverage_summary(enriched)
    report = render_report(coverage, client, queried)
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
