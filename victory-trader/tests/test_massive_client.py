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
