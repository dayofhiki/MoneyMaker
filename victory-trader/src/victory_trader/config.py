from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    massive_api_key: str


def load_settings() -> Settings:
    load_dotenv()
    api_key = os.getenv("MASSIVE_API_KEY", "").strip()
    if not api_key or api_key == "replace_me":
        raise RuntimeError(
            "MASSIVE_API_KEY is missing. Copy .env.example to .env and set your API key."
        )
    return Settings(massive_api_key=api_key)
