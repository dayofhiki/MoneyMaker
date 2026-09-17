from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    massive_api_key: str
    massive_s3_access_key: str | None = None
    massive_s3_secret_key: str | None = None


def load_settings() -> Settings:
    load_dotenv()
    api_key = os.getenv("MASSIVE_API_KEY", "").strip()
    if not api_key or api_key == "replace_me":
        raise RuntimeError(
            "MASSIVE_API_KEY is missing. Copy .env.example to .env and set your API key."
        )
    access_key = os.getenv("MASSIVE_S3_ACCESS_KEY", "").strip() or None
    secret_key = os.getenv("MASSIVE_S3_SECRET_KEY", "").strip() or None
    return Settings(
        massive_api_key=api_key,
        massive_s3_access_key=access_key,
        massive_s3_secret_key=secret_key,
    )


def require_flatfile_credentials(settings: Settings) -> tuple[str, str]:
    if not settings.massive_s3_access_key or not settings.massive_s3_secret_key:
        raise RuntimeError(
            "Massive Flat Files credentials are missing. Set MASSIVE_S3_ACCESS_KEY "
            "and MASSIVE_S3_SECRET_KEY."
        )
    return settings.massive_s3_access_key, settings.massive_s3_secret_key
