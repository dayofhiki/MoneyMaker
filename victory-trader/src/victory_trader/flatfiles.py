from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


FLATFILES_ENDPOINT = "https://files.massive.com"
FLATFILES_BUCKET = "flatfiles"
STOCKS_MINUTE_PREFIX = "us_stocks_sip/minute_aggs_v1"
STOCKS_DAY_PREFIX = "us_stocks_sip/day_aggs_v1"


def stock_flatfile_key(dataset_prefix: str, day: date) -> str:
    return (
        f"{dataset_prefix}/{day.year:04d}/{day.month:02d}/"
        f"{day.isoformat()}.csv.gz"
    )


@dataclass(frozen=True)
class FlatFileObject:
    dataset: str
    day: date
    key: str
    size_bytes: int | None


class MassiveFlatFilesClient:
    """Secret-safe S3-compatible client for Massive historical Flat Files."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        *,
        endpoint_url: str = FLATFILES_ENDPOINT,
        bucket: str = FLATFILES_BUCKET,
    ) -> None:
        if not access_key or not secret_key:
            raise ValueError("Flat Files access and secret keys must be non-empty")
        self.bucket = bucket
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._s3 = session.client(
            "s3",
            endpoint_url=endpoint_url,
            config=Config(signature_version="s3v4"),
        )

    def head(self, dataset_prefix: str, day: date) -> FlatFileObject:
        key = stock_flatfile_key(dataset_prefix, day)
        try:
            response = self._s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(f"Massive Flat File not found: {key}") from exc
            raise
        raw_size = response.get("ContentLength")
        size = int(raw_size) if raw_size is not None else None
        return FlatFileObject(
            dataset=dataset_prefix,
            day=day,
            key=key,
            size_bytes=size,
        )

    def download(
        self,
        dataset_prefix: str,
        day: date,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        key = stock_flatfile_key(dataset_prefix, day)
        if destination.exists() and not overwrite:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            self._s3.download_file(self.bucket, key, str(temporary))
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def check_stock_aggregate_access(self, day: date) -> tuple[FlatFileObject, FlatFileObject]:
        """Verify credentials and entitlement without downloading market data."""
        return (
            self.head(STOCKS_DAY_PREFIX, day),
            self.head(STOCKS_MINUTE_PREFIX, day),
        )
