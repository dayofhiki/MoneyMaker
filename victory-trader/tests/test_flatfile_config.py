import pytest

from victory_trader.config import load_flatfile_credentials


def test_flatfile_credentials_do_not_require_rest_api_key(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY", "access")
    monkeypatch.setenv("MASSIVE_S3_SECRET_KEY", "secret")

    assert load_flatfile_credentials() == ("access", "secret")


def test_flatfile_credentials_require_both_values(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY", "access")
    monkeypatch.delenv("MASSIVE_S3_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Flat Files credentials are missing"):
        load_flatfile_credentials()
