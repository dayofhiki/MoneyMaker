from pathlib import Path

import pytest
import requests

from victory_trader.massive_client import MassiveClient


class FakeResponse:
    def __init__(self, payload=None, status_code=200, headers=None):
        self.status_code = status_code
        self._payload = payload or {"status": "OK"}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error
        return None

    def json(self):
        return self._payload


def test_get_uses_authorization_header_not_query_key(monkeypatch):
    captured = {}

    def fake_get(url, *, params, headers, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("victory_trader.massive_client.requests.get", fake_get)
    client = MassiveClient("super-secret-key")
    payload = client._get("/v3/reference/tickers/AAPL", {"date": "2026-09-01"})

    assert payload == {"status": "OK"}
    assert captured["url"] == "https://api.massive.com/v3/reference/tickers/AAPL"
    assert captured["params"] == {"date": "2026-09-01"}
    assert "apiKey" not in captured["params"]
    assert captured["headers"]["Authorization"] == "Bearer super-secret-key"


def test_massive_cache_reuses_successful_response_without_storing_secret_or_sleeping(tmp_path, monkeypatch):
    calls = []
    sleeps = []

    def fake_get(url, *, params, headers, timeout):
        calls.append(url)
        return FakeResponse({"results": [{"ticker": "AAA"}]})

    monkeypatch.setattr("victory_trader.massive_client.requests.get", fake_get)
    monkeypatch.setattr("victory_trader.massive_client.time.sleep", lambda seconds: sleeps.append(seconds))

    cache_dir = tmp_path / "massive"
    client = MassiveClient("super-secret-key", cache_dir=cache_dir, request_interval_seconds=12.5)
    first = client._get("/v3/reference/tickers/AAA", {"date": "2026-09-15"})
    second = client._get("/v3/reference/tickers/AAA", {"date": "2026-09-15"})

    assert first == second
    assert len(calls) == 1
    assert sleeps == []
    assert client.stats.network_requests == 1
    assert client.stats.cache_hits == 1

    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    assert "super-secret-key" not in cache_files[0].read_text(encoding="utf-8")
    assert "super-secret-key" not in cache_files[0].name


def test_429_retries_using_retry_after(monkeypatch):
    responses = [
        FakeResponse(status_code=429, headers={"Retry-After": "0"}),
        FakeResponse({"status": "OK"}),
    ]
    sleeps = []

    monkeypatch.setattr("victory_trader.massive_client.requests.get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("victory_trader.massive_client.time.sleep", lambda seconds: sleeps.append(seconds))
    client = MassiveClient("secret", max_retries=2)

    assert client._get("/test", use_cache=False) == {"status": "OK"}
    assert client.stats.network_requests == 2
    assert client.stats.retries == 1
    assert sleeps == [0.0]


def test_auth_failure_is_fatal_without_retry(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse(status_code=401)

    monkeypatch.setattr("victory_trader.massive_client.requests.get", fake_get)
    client = MassiveClient("bad-secret", max_retries=4)
    with pytest.raises(requests.HTTPError):
        client._get("/private", use_cache=False)
    assert len(calls) == 1
    assert client.stats.retries == 0


def test_historical_minute_bars_default_to_unadjusted(monkeypatch):
    captured = {}

    def fake_get(url, *, params, headers, timeout):
        captured["params"] = params
        return FakeResponse({"results": []})

    monkeypatch.setattr("victory_trader.massive_client.requests.get", fake_get)
    client = MassiveClient("secret")
    from datetime import date

    client.minute_bars("AAPL", date(2026, 9, 15))
    assert captured["params"]["adjusted"] == "false"


def test_latest_previous_close_is_not_persistently_cached(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, *, params, headers, timeout):
        calls.append(url)
        return FakeResponse({"results": [{"c": 123.0}]})

    monkeypatch.setattr("victory_trader.massive_client.requests.get", fake_get)
    client = MassiveClient("secret", cache_dir=Path(tmp_path))
    client.previous_close("AAPL")
    client.previous_close("AAPL")

    assert len(calls) == 2
    assert list(Path(tmp_path).glob("*.json")) == []


def test_news_uses_timestamp_bounds_and_paginates(monkeypatch):
    calls = []
    payloads = [
        {
            "results": [{"id": "a", "published_utc": "2026-01-02T14:00:00Z"}],
            "next_url": "https://api.massive.com/v2/reference/news?cursor=next",
        },
        {
            "results": [{"id": "b", "published_utc": "2026-01-02T14:30:00Z"}],
        },
    ]

    def fake_get(url, *, params, headers, timeout):
        calls.append((url, params))
        return FakeResponse(payloads.pop(0))

    monkeypatch.setattr("victory_trader.massive_client.requests.get", fake_get)
    client = MassiveClient("secret")
    rows = client.news(
        "AAPL",
        published_gte="2026-01-01T15:00:00Z",
        published_lte="2026-01-02T15:00:00Z",
    )

    assert [row["id"] for row in rows] == ["a", "b"]
    assert calls[0][1]["ticker"] == "AAPL"
    assert calls[0][1]["published_utc.gte"] == "2026-01-01T15:00:00Z"
    assert calls[0][1]["published_utc.lte"] == "2026-01-02T15:00:00Z"
    assert calls[1][1] == {"cursor": "next"}


def test_quotes_use_nanosecond_timestamp_bounds_and_paginate(monkeypatch):
    calls = []
    payloads = [
        {
            "results": [
                {
                    "bid_price": 4.99,
                    "ask_price": 5.01,
                    "sip_timestamp": 1760000000000000000,
                }
            ],
            "next_url": "https://api.massive.com/v3/quotes/AAA?cursor=next",
        },
        {
            "results": [
                {
                    "bid_price": 5.00,
                    "ask_price": 5.02,
                    "sip_timestamp": 1760000001000000000,
                }
            ],
        },
    ]

    def fake_get(url, *, params, headers, timeout):
        calls.append((url, params))
        return FakeResponse(payloads.pop(0))

    monkeypatch.setattr("victory_trader.massive_client.requests.get", fake_get)
    client = MassiveClient("secret")
    rows = client.quotes(
        "AAA",
        timestamp_gte=1760000000000000000,
        timestamp_lte=1760000002000000000,
    )

    assert len(rows) == 2
    assert calls[0][0].endswith("/v3/quotes/AAA")
    assert calls[0][1]["timestamp.gte"] == 1760000000000000000
    assert calls[0][1]["timestamp.lte"] == 1760000002000000000
    assert calls[1][1] == {"cursor": "next"}
