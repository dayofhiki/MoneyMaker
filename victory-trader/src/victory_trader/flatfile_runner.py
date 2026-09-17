from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .config import load_settings, require_flatfile_credentials
from .flatfile_dataset import build_flatfile_market_event_dataset
from .flatfiles import MassiveFlatFileStore, MassiveFlatFilesClient
from .market_calendar import is_us_equity_trading_day
from .market_dataset import DayBuildStats, save_dataset
from .massive_client import MassiveClient
from .multi_day import daterange


REST_CACHE_DIR = Path("data/cache/massive")
FLATFILE_CACHE_DIR = Path("data/cache/massive-flatfiles")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0) - before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def build_flatfile_range(
    start: date,
    end: date,
    *,
    output: Path,
    checkpoint_dir: Path,
    manifest_path: Path,
    min_price: float = 0.50,
    max_price: float = 20.0,
    min_high_return_pct: float = 10.0,
    max_candidates: int | None = None,
    historical_context_days: int = 35,
    annotate_halts: bool = False,
) -> pd.DataFrame:
    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    rest_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=REST_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    flat_client = MassiveFlatFilesClient(access_key, secret_key)
    store = MassiveFlatFileStore(flat_client, FLATFILE_CACHE_DIR)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    days_manifest: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()

    for day in daterange(start, end):
        status_path = checkpoint_dir / f"{day.isoformat()}.json"
        data_path = checkpoint_dir / f"{day.isoformat()}.parquet"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") == "complete":
                if status.get("kind") == "events":
                    if not data_path.exists():
                        raise RuntimeError(f"Flat File checkpoint data missing for {day}")
                    frames.append(pd.read_parquet(data_path))
                resumed = dict(status)
                resumed["resumed"] = True
                days_manifest.append(resumed)
                continue

        if not is_us_equity_trading_day(day):
            status = {
                "state": "complete",
                "kind": "closed_market",
                "trading_day": day.isoformat(),
                "backend": "massive_flatfiles",
            }
            _atomic_json(status_path, status)
            days_manifest.append(status)
            continue

        rest_before = rest_client.stats.to_dict()
        flat_before = store.stats.to_dict()
        stats = DayBuildStats(trading_day=day.isoformat())
        try:
            frame = build_flatfile_market_event_dataset(
                rest_client,
                store,
                day,
                min_price=min_price,
                max_price=max_price,
                min_high_return_pct=min_high_return_pct,
                max_candidates=max_candidates,
                historical_context_days=historical_context_days,
                annotate_halts=annotate_halts,
                build_stats=stats,
            )
        except Exception as exc:
            status = {
                "state": "error",
                "kind": "error",
                "trading_day": day.isoformat(),
                "backend": "massive_flatfiles",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "build_stats": stats.to_dict(),
                "rest_stats_delta": _delta(rest_before, rest_client.stats.to_dict()),
                "flatfile_stats_delta": _delta(flat_before, store.stats.to_dict()),
            }
            days_manifest.append(status)
            _atomic_json(
                manifest_path,
                {
                    "schema_version": 2,
                    "backend": "massive_flatfiles",
                    "started_at_utc": started_at,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "days": days_manifest,
                    "complete": False,
                },
            )
            raise

        kind = "no_events" if frame.empty else "events"
        if not frame.empty:
            save_dataset(frame, data_path)
            frames.append(frame)
        status = {
            "state": "complete",
            "kind": kind,
            "trading_day": day.isoformat(),
            "backend": "massive_flatfiles",
            "build_stats": stats.to_dict(),
            "rest_stats_delta": _delta(rest_before, rest_client.stats.to_dict()),
            "flatfile_stats_delta": _delta(flat_before, store.stats.to_dict()),
        }
        _atomic_json(status_path, status)
        days_manifest.append(status)

    result = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["trading_day", "ticker", "timestamp_ms", "threshold_pct"])
        .reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )
    if not result.empty:
        save_dataset(result, output)

    _atomic_json(
        manifest_path,
        {
            "schema_version": 2,
            "backend": "massive_flatfiles",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days": days_manifest,
            "rest_stats": rest_client.stats.to_dict(),
            "flatfile_stats": store.stats.to_dict(),
            "event_rows": int(len(result)),
            "event_tickers": int(result["ticker"].nunique()) if not result.empty else 0,
            "complete": True,
        },
    )
    print(f"REST stats: {rest_client.stats.to_dict()}")
    print(f"Flat File stats: {store.stats.to_dict()}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.flatfile_runner",
        description="Build MoneyMaker historical research datasets from Massive Flat Files.",
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints-flatfiles"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/flatfile-manifest.json"))
    parser.add_argument("--min-price", type=float, default=0.50)
    parser.add_argument("--max-price", type=float, default=20.0)
    parser.add_argument("--min-high-return", type=float, default=10.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--history-days", type=int, default=35)
    parser.add_argument("--annotate-halts", action="store_true")
    args = parser.parse_args()

    result = build_flatfile_range(
        args.start,
        args.end,
        output=args.output,
        checkpoint_dir=args.checkpoint_dir,
        manifest_path=args.manifest,
        min_price=args.min_price,
        max_price=args.max_price,
        min_high_return_pct=args.min_high_return,
        max_candidates=args.max_candidates,
        historical_context_days=args.history_days,
        annotate_halts=args.annotate_halts,
    )
    if result.empty:
        print("No qualifying momentum events found in requested Flat File range.")
    else:
        print(
            f"Saved {len(result)} event rows across {result['ticker'].nunique()} tickers "
            f"and {result['trading_day'].nunique()} trading days to {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
