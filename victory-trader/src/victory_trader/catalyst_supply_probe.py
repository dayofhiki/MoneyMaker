from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from .analytics import load_event_dataset
from .candidate_v2_entry_research import SIGNAL_CONFIG, _score_signal_only
from .candidate_v2_research import _fit_threshold_params
from .config import load_settings
from .massive_client import MassiveClient


SAMPLE_PER_MONTH = 10
NEWS_WINDOW_HOURS = 24


def _iso_utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _sentiment_counts(articles: list[dict]) -> tuple[int, int, int]:
    positive = negative = neutral = 0
    for article in articles:
        for insight in article.get("insights") or []:
            sentiment = str(insight.get("sentiment") or "").lower()
            if sentiment == "positive":
                positive += 1
            elif sentiment == "negative":
                negative += 1
            elif sentiment == "neutral":
                neutral += 1
    return positive, negative, neutral


def _sample_signals(monthly_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [frame for month, frame in monthly_frames.items() if month != holdout_month],
            ignore_index=True,
        )
        params = _fit_threshold_params(train, SIGNAL_CONFIG)
        scored = _score_signal_only(holdout, params)
        selected = scored.loc[scored["candidate_v2_signal"].astype(bool)].copy()
        selected["month"] = holdout_month
        selected = selected.sort_values(
            ["trading_day", "threshold_pct", "ticker", "timestamp_ms"],
            kind="stable",
        ).head(SAMPLE_PER_MONTH)
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def run_probe(
    monthly_frames: dict[str, pd.DataFrame],
    client: MassiveClient,
) -> pd.DataFrame:
    sample = _sample_signals(monthly_frames)
    rows = []

    for _, event in sample.iterrows():
        ticker = str(event["ticker"]).upper()
        event_ms = int(event["timestamp_ms"])
        event_dt = datetime.fromtimestamp(event_ms / 1000.0, tz=UTC)
        start_dt = event_dt - timedelta(hours=NEWS_WINDOW_HOURS)

        articles = client.news(
            ticker,
            published_gte=start_dt.isoformat().replace("+00:00", "Z"),
            published_lte=event_dt.isoformat().replace("+00:00", "Z"),
        )
        articles = [
            article
            for article in articles
            if str(article.get("published_utc") or "") <= _iso_utc(event_ms)
        ]

        positive, negative, neutral = _sentiment_counts(articles)
        latest_age_minutes = None
        latest_title = None
        if articles:
            latest = max(articles, key=lambda x: str(x.get("published_utc") or ""))
            published = datetime.fromisoformat(
                str(latest["published_utc"]).replace("Z", "+00:00")
            )
            latest_age_minutes = (event_dt - published).total_seconds() / 60.0
            latest_title = latest.get("title")

        details_payload = client.ticker_details(ticker, event_dt.date())
        details = details_payload.get("results") or {}

        rows.append(
            {
                "month": event["month"],
                "trading_day": str(event["trading_day"]),
                "ticker": ticker,
                "threshold_pct": float(event["threshold_pct"]),
                "event_utc": _iso_utc(event_ms),
                "news_count_24h": len(articles),
                "news_count_6h": sum(
                    1
                    for article in articles
                    if (
                        event_dt
                        - datetime.fromisoformat(
                            str(article["published_utc"]).replace("Z", "+00:00")
                        )
                    ).total_seconds()
                    <= 6 * 3600
                ),
                "positive_insights": positive,
                "negative_insights": negative,
                "neutral_insights": neutral,
                "latest_news_age_minutes": latest_age_minutes,
                "latest_news_title": latest_title,
                "details_market_cap": details.get("market_cap"),
                "details_weighted_shares_outstanding": details.get(
                    "weighted_shares_outstanding"
                ),
                "details_share_class_shares_outstanding": details.get(
                    "share_class_shares_outstanding"
                ),
                "details_cik": details.get("cik"),
                "return_15m_pct": event.get("return_15m_pct"),
                "return_15m_base_net_return_pct": event.get(
                    "return_15m_base_net_return_pct"
                ),
            }
        )

    return pd.DataFrame(rows)


def render_report(frame: pd.DataFrame, client: MassiveClient) -> str:
    if frame.empty:
        return "=== MoneyMaker Catalyst Probe ===\nNo sampled candidate signals."

    month_summary = (
        frame.assign(
            has_news_24h=frame["news_count_24h"].gt(0),
            has_news_6h=frame["news_count_6h"].gt(0),
            has_sentiment=(frame["positive_insights"] + frame["negative_insights"]).gt(0),
            has_market_cap=frame["details_market_cap"].notna(),
            has_weighted_shares=frame["details_weighted_shares_outstanding"].notna(),
        )
        .groupby("month")
        .agg(
            sampled=("ticker", "size"),
            news_24h_coverage=("has_news_24h", "mean"),
            news_6h_coverage=("has_news_6h", "mean"),
            sentiment_coverage=("has_sentiment", "mean"),
            market_cap_coverage=("has_market_cap", "mean"),
            weighted_shares_coverage=("has_weighted_shares", "mean"),
        )
        .reset_index()
    )

    overall_news = float(frame["news_count_24h"].gt(0).mean())
    overall_6h = float(frame["news_count_6h"].gt(0).mean())
    detail_market_cap = float(frame["details_market_cap"].notna().mean())
    weighted_shares = float(
        frame["details_weighted_shares_outstanding"].notna().mean()
    )

    display = frame[
        [
            "month",
            "trading_day",
            "ticker",
            "threshold_pct",
            "news_count_24h",
            "news_count_6h",
            "positive_insights",
            "negative_insights",
            "latest_news_age_minutes",
            "details_market_cap",
            "details_weighted_shares_outstanding",
            "latest_news_title",
        ]
    ]

    return "\n".join(
        [
            "=== MoneyMaker Catalyst / Supply Access Probe ===",
            f"sampled_events={len(frame)}",
            f"unique_tickers={frame['ticker'].nunique()}",
            f"news_24h_coverage={overall_news:.4f}",
            f"news_6h_coverage={overall_6h:.4f}",
            f"market_cap_field_coverage={detail_market_cap:.4f}",
            f"weighted_shares_field_coverage={weighted_shares:.4f}",
            f"Massive_client_stats={client.stats.to_dict()}",
            "",
            "=== Coverage by held-out month ===",
            month_summary.to_string(index=False),
            "",
            "=== Sampled events ===",
            display.to_string(index=False),
            "",
            "Safety note: news uses published_utc <= event_utc. Point-in-time ticker details are audited here but are not yet approved as training features until filing/effective timestamp semantics are verified.",
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.catalyst_supply_probe"
    )
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
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
    frames = {label: load_event_dataset(path) for label, path in args.dataset}
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
