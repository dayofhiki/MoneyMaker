from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

FLATFILES_ENDPOINT = "https://files.massive.com"
FLATFILES_BUCKET = "flatfiles"
STOCKS_MINUTE_PREFIX = "us_stocks_sip/minute_aggs_v1"
STOCKS_DAY_PREFIX = "us_stocks_sip/day_aggs_v1"
FLATFILE_REQUIRED_COLUMNS = (
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
)


def stock_flatfile_key(dataset_prefix: str, day: date) -> str:
    return (
        f"{dataset_prefix}/{day.year:04d}/{day.month:02d}/"
        f"{day.isoformat()}.csv.gz"
    )


def _cache_stem(cache_dir: Path, dataset_prefix: str, day: date) -> Path:
    return (
        cache_dir
        / dataset_prefix
        / f"{day.year:04d}"
        / f"{day.month:02d}"
        / day.isoformat()
    )


@dataclass(frozen=True)
class FlatFileObject:
    dataset: str
    day: date
    key: str
    size_bytes: int | None


@dataclass
class FlatFileStoreStats:
    cache_hits: int = 0
    downloads: int = 0
    downloaded_bytes: int = 0
    parquet_writes: int = 0
    parquet_reads: int = 0

    def to_dict(self) -> dict[str, int]:
        return vars(self).copy()


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


@dataclass
class MassiveFlatFileStore:
    """Download each market-wide daily file once, then query it locally as Parquet."""

    client: MassiveFlatFilesClient
    cache_dir: Path
    stats: FlatFileStoreStats = field(default_factory=FlatFileStoreStats)

    def _paths(self, dataset_prefix: str, day: date) -> tuple[Path, Path]:
        stem = _cache_stem(self.cache_dir, dataset_prefix, day)
        return (
            stem.parent / f"{stem.name}.csv.gz",
            stem.parent / f"{stem.name}.parquet",
        )

    def ensure_parquet(self, dataset_prefix: str, day: date) -> Path:
        raw_path, parquet_path = self._paths(dataset_prefix, day)
        if parquet_path.exists():
            self.stats.cache_hits += 1
            return parquet_path

        existed_before = raw_path.exists()
        self.client.download(dataset_prefix, day, raw_path)
        if not existed_before:
            self.stats.downloads += 1
            self.stats.downloaded_bytes += int(raw_path.stat().st_size)

        frame = pd.read_csv(raw_path, compression="gzip")
        missing = set(FLATFILE_REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Massive Flat File missing required columns: {sorted(missing)}"
            )

        frame = frame.loc[:, list(FLATFILE_REQUIRED_COLUMNS)].copy()
        frame["ticker"] = frame["ticker"].astype("string").str.upper()
        for column in ("volume", "open", "close", "high", "low"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame["window_start"] = pd.to_numeric(
            frame["window_start"], errors="raise"
        ).astype("int64")
        frame["transactions"] = pd.to_numeric(
            frame["transactions"], errors="coerce"
        )

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = parquet_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(parquet_path)
        self.stats.parquet_writes += 1
        raw_path.unlink(missing_ok=True)
        return parquet_path

    def read(
        self,
        dataset_prefix: str,
        day: date,
        *,
        tickers: set[str] | None = None,
    ) -> pd.DataFrame:
        parquet_path = self.ensure_parquet(dataset_prefix, day)
        filters = None
        if tickers is not None:
            normalized = sorted({ticker.upper() for ticker in tickers})
            if not normalized:
                return pd.DataFrame(columns=FLATFILE_REQUIRED_COLUMNS)
            filters = [("ticker", "in", normalized)]
        frame = pd.read_parquet(parquet_path, filters=filters)
        self.stats.parquet_reads += 1
        return frame.reset_index(drop=True)

    def day_aggregates(self, day: date) -> pd.DataFrame:
        return self.read(STOCKS_DAY_PREFIX, day)

    def minute_aggregates(
        self,
        day: date,
        *,
        tickers: set[str] | None = None,
    ) -> pd.DataFrame:
        frame = self.read(STOCKS_MINUTE_PREFIX, day, tickers=tickers)
        if frame.empty:
            return pd.DataFrame(columns=("ticker", "t", "o", "h", "l", "c", "v", "n"))
        result = frame.rename(
            columns={
                "volume": "v",
                "open": "o",
                "close": "c",
                "high": "h",
                "low": "l",
                "transactions": "n",
            }
        ).copy()
        result["t"] = (result["window_start"].astype("int64") // 1_000_000).astype("int64")
        return result.loc[:, ["ticker", "t", "o", "h", "l", "c", "v", "n"]]

    def grouped_daily_payload(self, day: date) -> dict:
        frame = self.day_aggregates(day)
        results = []
        for row in frame.itertuples(index=False):
            results.append(
                {
                    "T": str(row.ticker),
                    "v": float(row.volume),
                    "o": float(row.open),
                    "c": float(row.close),
                    "h": float(row.high),
                    "l": float(row.low),
                    "n": None if pd.isna(row.transactions) else float(row.transactions),
                    "t": int(row.window_start // 1_000_000),
                }
            )
        return {"results": results}
