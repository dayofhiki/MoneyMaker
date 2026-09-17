from victory_trader.massive_client import MassiveClient


class FakeResponse:
    def __init__(self):
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"status": "OK"}


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
