from datetime import date, datetime
from zoneinfo import ZoneInfo

from victory_trader.market_dataset import build_market_event_dataset
from victory_trader.massive_client import MassiveClient


ET = ZoneInfo("America/New_York")


def ts(hour: int, minute: int) -> int:
    return int(datetime(2026, 9, 15, hour, minute, tzinfo=ET).timestamp() * 1000)


class FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_reference_tickers_paginates_without_forwarding_api_key(monkeypatch):
    calls = []
    responses = [
        {
            "results": [{"ticker": "AAA"}],
            "next_url": "https://api.massive.com/v3/reference/tickers?cursor=abc&apiKey=must-not-forward",
        },
        {"results": [{"ticker": "BBB"}]},
    ]

    def fake_get(url, *, params, headers, timeout):
        calls.append((url, dict(params), dict(headers)))
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr("victory_trader.massive_client.requests.get", fake_get)
    client = MassiveClient("real-secret")
    rows = client.reference_tickers(date(2026, 9, 15))

    assert [row["ticker"] for row in rows] == ["AAA", "BBB"]
    assert len(calls) == 2
    assert calls[0][1]["limit"] == 1000
    assert calls[1][1] == {"cursor": "abc"}
    assert all("apiKey" not in params for _, params, _ in calls)
    assert all(headers["Authorization"] == "Bearer real-secret" for _, _, headers in calls)


class BulkClient:
    def __init__(self):
        self.market = {
            "2026-09-14": {"results": [{"T": "AAA", "c": 2.0}]},
            "2026-09-15": {
                "results": [{"T": "AAA", "h": 3.2, "c": 2.8, "v": 1_000_000, "vw": 2.6}]
            },
        }
        self.bulk_calls = 0

    def _get(self, path, params=None):
        return self.market.get(path.rsplit("/", 1)[-1], {"results": []})

    def splits_on(self, day):
        return {"results": []}

    def reference_tickers(self, day, *, market="stocks", security_type="CS", active=True):
        self.bulk_calls += 1
        return [
            {
                "ticker": "AAA",
                "name": "AAA Corp",
                "type": "CS",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
                "active": True,
            }
        ]

    def ticker_details(self, ticker, day=None):
        raise AssertionError("per-ticker metadata endpoint should not be used in formal bulk path")

    def minute_bars(self, ticker, day):
        return {
            "results": [
                {"t": ts(9, 30), "o": 2.00, "h": 2.05, "l": 1.95, "c": 2.00, "v": 1000},
                {"t": ts(9, 31), "o": 2.00, "h": 2.45, "l": 2.00, "c": 2.40, "v": 5000},
                {"t": ts(9, 32), "o": 2.40, "h": 2.65, "l": 2.35, "c": 2.60, "v": 6000},
                {"t": ts(9, 33), "o": 2.60, "h": 3.05, "l": 2.55, "c": 3.00, "v": 8000},
                {"t": ts(9, 34), "o": 3.00, "h": 3.20, "l": 2.90, "c": 3.10, "v": 9000},
                {"t": ts(9, 35), "o": 3.10, "h": 3.15, "l": 3.00, "c": 3.05, "v": 7000},
            ]
        }


def test_market_dataset_uses_one_bulk_taxonomy_lookup():
    client = BulkClient()
    result = build_market_event_dataset(
        client,
        date(2026, 9, 15),
        thresholds_pct=(20, 30, 50),
        min_high_return_pct=20,
        horizons=(1, 2),
        request_interval_seconds=0,
    )

    assert not result.empty
    assert client.bulk_calls == 1
    assert set(result["security_type"]) == {"CS"}
    assert set(result["primary_exchange"]) == {"XNAS"}
