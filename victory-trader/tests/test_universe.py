from victory_trader.universe import metadata_from_payload, split_tickers_from_payload


def test_common_stock_metadata_is_accepted():
    metadata = metadata_from_payload(
        {
            "results": {
                "ticker": "AAA",
                "name": "AAA Corp",
                "type": "CS",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
                "active": True,
                "market_cap": 42_000_000,
                "share_class_shares_outstanding": 8_000_000,
            }
        }
    )
    assert metadata.is_research_common_stock
    assert metadata.market_cap == 42_000_000.0


def test_etf_and_wrong_exchange_are_rejected():
    etf = metadata_from_payload(
        {"results": {"ticker": "ETF1", "type": "ETF", "market": "stocks", "locale": "us", "primary_exchange": "XNAS"}}
    )
    arca = metadata_from_payload(
        {"results": {"ticker": "ARCA", "type": "CS", "market": "stocks", "locale": "us", "primary_exchange": "ARCX"}}
    )
    assert not etf.is_research_common_stock
    assert not arca.is_research_common_stock


def test_split_ticker_extraction():
    result = split_tickers_from_payload(
        {"results": [{"ticker": "AAA"}, {"ticker": "bbb"}, {"ticker": None}]}
    )
    assert result == {"AAA", "BBB"}
