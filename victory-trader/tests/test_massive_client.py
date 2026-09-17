from pathlib import Path

from victory_trader.massive_client import MassiveClient


class FakeResponse:
    def __init__(self, payload=None):
        self.status_code = 200
        self._payload = payload or {"status": "OK"}

    def raise_for_status(self):
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


def test_massive_cache_reuses_successful_response_without_storing_secret(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, *, params, headers, timeout):
        calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeResponse({"results": [{"ticker": "AAA"}]})

    monkeypatch.setattr("victory_trader.massive_client.requests.get", fake_get)

    cache_dir = tmp_path / "massive"
    client = MassiveClient("super-secret-key", cache_dir=cache_dir)

    first = client._get("/v3/reference/tickers/AAA", {"date": "2026-09-15"})
    second = client._get("/v3/reference/tickers/AAA", {"date": "2026-09-15"})

    assert first == second
    assert len(calls) == 1
    assert calls[0]["params"] == {"date": "2026-09-15"}
    assert calls[0]["headers"] == {"Authorization": "Bearer super-secret-key"}

    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    assert "super-secret-key" not in cache_files[0].read_text(encoding="utf-8")
    assert "super-secret-key" not in cache_files[0].name


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
