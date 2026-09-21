from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from victory_trader.config import load_settings
from victory_trader.massive_client import MassiveClient


LOOKBACK_HOURS = 72
RAW_NEWS_COLUMNS = (
    "news_query_success",
    "news_strict_prior",
    "news_count_6h",
    "news_count_24h",
    "news_count_72h",
    "news_latest_age_minutes",
    "news_positive_insights_72h",
    "news_negative_insights_72h",
    "news_neutral_insights_72h",
)


def _utc_timestamp(value: object) -> pd.Timestamp | None:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def _iso_utc(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def fetch_news_history(
    frames: dict[str, pd.DataFrame],
    client: MassiveClient,
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    tickers: set[str] = set()
    timestamps: list[int] = []
    for frame in frames.values():
        tickers.update(frame["ticker"].astype(str).str.upper().unique())
        timestamps.extend(
            pd.to_numeric(frame["t"], errors="coerce")
            .dropna()
            .astype("int64")
            .tolist()
        )
    if not timestamps:
        return {}, set()

    earliest = pd.Timestamp(min(timestamps), unit="ms", tz="UTC")
    latest = pd.Timestamp(max(timestamps), unit="ms", tz="UTC")
    start = earliest - pd.Timedelta(hours=LOOKBACK_HOURS)

    history: dict[str, list[dict[str, object]]] = {}
    queried: set[str] = set()
    for ticker in sorted(tickers):
        rows = client.news(
            ticker,
            published_gte=_iso_utc(start),
            published_lte=_iso_utc(latest),
        )
        cleaned: list[dict[str, object]] = []
        for row in rows:
            published = _utc_timestamp(row.get("published_utc"))
            if published is None:
                continue
            item = dict(row)
            item["_published_utc"] = published
            cleaned.append(item)
        cleaned.sort(key=lambda row: row["_published_utc"])
        history[ticker] = cleaned
        queried.add(ticker)
    return history, queried


def _ticker_sentiments(
    articles: list[dict[str, object]], ticker: str
) -> tuple[int, int, int]:
    positive = negative = neutral = 0
    for article in articles:
        for insight in article.get("insights") or []:
            insight_ticker = str(insight.get("ticker") or "").upper()
            if insight_ticker and insight_ticker != ticker:
                continue
            sentiment = str(insight.get("sentiment") or "").lower()
            if sentiment == "positive":
                positive += 1
            elif sentiment == "negative":
                negative += 1
            elif sentiment == "neutral":
                neutral += 1
    return positive, negative, neutral


def news_features(
    articles: list[dict[str, object]],
    *,
    ticker: str,
    event_utc: pd.Timestamp,
) -> dict[str, float]:
    lower = event_utc - pd.Timedelta(hours=LOOKBACK_HOURS)
    prior = [
        row
        for row in articles
        if lower <= row["_published_utc"] < event_utc
    ]
    if any(row["_published_utc"] >= event_utc for row in prior):
        raise ValueError("news point-in-time violation")

    ages = [
        (event_utc - row["_published_utc"]).total_seconds() / 60.0
        for row in prior
    ]
    positive, negative, neutral = _ticker_sentiments(prior, ticker)
    return {
        "news_query_success": 1.0,
        "news_strict_prior": 1.0,
        "news_count_6h": float(sum(age <= 6 * 60 for age in ages)),
        "news_count_24h": float(sum(age <= 24 * 60 for age in ages)),
        "news_count_72h": float(len(prior)),
        "news_latest_age_minutes": min(ages) if ages else np.nan,
        "news_positive_insights_72h": float(positive),
        "news_negative_insights_72h": float(negative),
        "news_neutral_insights_72h": float(neutral),
    }


def enrich_frame(
    frame: pd.DataFrame,
    history: dict[str, list[dict[str, object]]],
    queried: set[str],
) -> pd.DataFrame:
    result = frame.copy()
    rows: list[dict[str, float]] = []
    for item in result.itertuples(index=False):
        ticker = str(item.ticker).upper()
        if ticker not in queried:
            raise ValueError(f"news query missing for ticker: {ticker}")
        event = pd.Timestamp(int(item.t), unit="ms", tz="UTC")
        rows.append(
            news_features(
                history.get(ticker, []),
                ticker=ticker,
                event_utc=event,
            )
        )
    features = pd.DataFrame(rows, index=result.index)
    for column in RAW_NEWS_COLUMNS:
        result[column] = features[column]
    return result


def _coverage(label: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "month": label,
        "anchors": int(len(frame)),
        "query_success": float(frame["news_query_success"].eq(1.0).mean()),
        "strict_prior": bool(frame["news_strict_prior"].eq(1.0).all()),
        "news_6h_rate": float(frame["news_count_6h"].gt(0).mean()),
        "news_24h_rate": float(frame["news_count_24h"].gt(0).mean()),
        "news_72h_rate": float(frame["news_count_72h"].gt(0).mean()),
        "sentiment_rate": float(
            (
                frame["news_positive_insights_72h"]
                + frame["news_negative_insights_72h"]
                + frame["news_neutral_insights_72h"]
            ).gt(0).mean()
        ),
    }


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/cache/massive-rest")
    )
    args = parser.parse_args()

    frames = {label: pd.read_parquet(path) for label, path in args.dataset}
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )
    history, queried = fetch_news_history(frames, client)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = []
    for label, frame in frames.items():
        enriched = enrich_frame(frame, history, queried)
        enriched.to_parquet(
            args.output_dir / f"news-anchors-{label}.parquet",
            index=False,
            compression="zstd",
        )
        coverage.append(_coverage(label, enriched))

    coverage_frame = pd.DataFrame(coverage)
    report = "\n".join(
        [
            "=== MoneyMaker strictly-prior news enrichment v2.9 ===",
            "rule=published_utc strictly before anchor timestamp; lookback=72h",
            f"client_stats={client.stats.to_dict()}",
            "",
            coverage_frame.to_string(index=False),
        ]
    )
    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    coverage_frame.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
