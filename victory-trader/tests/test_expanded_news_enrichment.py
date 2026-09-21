import pandas as pd

from victory_trader.expanded_news_enrichment import (
    enrich_frame,
    news_features,
)


def _article(timestamp, ticker="AAA", sentiment="positive"):
    return {
        "_published_utc": pd.Timestamp(timestamp, tz="UTC"),
        "insights": [{"ticker": ticker, "sentiment": sentiment}],
    }


def test_news_features_are_strictly_prior_and_ticker_specific():
    event = pd.Timestamp("2026-01-02T15:00:00Z")
    rows = [
        _article("2026-01-02T14:00:00", sentiment="positive"),
        _article("2026-01-02T12:00:00", ticker="BBB", sentiment="negative"),
        _article("2026-01-02T15:00:00", sentiment="negative"),
    ]
    result = news_features(rows, ticker="AAA", event_utc=event)
    assert result["news_count_6h"] == 2.0
    assert result["news_count_72h"] == 2.0
    assert result["news_positive_insights_72h"] == 1.0
    assert result["news_negative_insights_72h"] == 0.0
    assert result["news_latest_age_minutes"] == 60.0


def test_enrichment_requires_completed_query():
    frame = pd.DataFrame(
        [{"ticker": "AAA", "t": 1767366000000, "trading_day": "2026-01-02"}]
    )
    enriched = enrich_frame(frame, {"AAA": []}, {"AAA"})
    assert enriched.iloc[0]["news_query_success"] == 1.0
    assert enriched.iloc[0]["news_count_72h"] == 0.0
