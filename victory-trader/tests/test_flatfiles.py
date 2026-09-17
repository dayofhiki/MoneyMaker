from datetime import date

from victory_trader.flatfiles import (
    STOCKS_DAY_PREFIX,
    STOCKS_MINUTE_PREFIX,
    MassiveFlatFilesClient,
    stock_flatfile_key,
)


class FakeS3:
    def __init__(self) -> None:
        self.head_calls = []

    def head_object(self, *, Bucket, Key):
        self.head_calls.append((Bucket, Key))
        return {"ContentLength": 12345}


class FakeSession:
    def __init__(self, fake_s3: FakeS3, **kwargs) -> None:
        self.fake_s3 = fake_s3
        self.kwargs = kwargs

    def client(self, service_name, *, endpoint_url, config):
        assert service_name == "s3"
        assert endpoint_url == "https://files.massive.com"
        assert config.signature_version == "s3v4"
        return self.fake_s3


def test_stock_flatfile_key_uses_massive_daily_layout():
    day = date(2026, 4, 1)
    assert stock_flatfile_key(STOCKS_MINUTE_PREFIX, day) == (
        "us_stocks_sip/minute_aggs_v1/2026/04/2026-04-01.csv.gz"
    )
    assert stock_flatfile_key(STOCKS_DAY_PREFIX, day) == (
        "us_stocks_sip/day_aggs_v1/2026/04/2026-04-01.csv.gz"
    )


def test_check_stock_aggregate_access_heads_both_files(monkeypatch):
    fake_s3 = FakeS3()
    captured = {}

    def session_factory(**kwargs):
        captured.update(kwargs)
        return FakeSession(fake_s3, **kwargs)

    monkeypatch.setattr("victory_trader.flatfiles.boto3.Session", session_factory)

    client = MassiveFlatFilesClient("ACCESS", "SECRET")
    day_obj, minute_obj = client.check_stock_aggregate_access(date(2026, 4, 1))

    assert captured == {
        "aws_access_key_id": "ACCESS",
        "aws_secret_access_key": "SECRET",
    }
    assert day_obj.size_bytes == 12345
    assert minute_obj.size_bytes == 12345
    assert fake_s3.head_calls == [
        (
            "flatfiles",
            "us_stocks_sip/day_aggs_v1/2026/04/2026-04-01.csv.gz",
        ),
        (
            "flatfiles",
            "us_stocks_sip/minute_aggs_v1/2026/04/2026-04-01.csv.gz",
        ),
    ]
