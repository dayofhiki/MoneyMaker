from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from .config import load_settings
from .market_calendar import is_us_equity_trading_day
from .market_dataset import DayBuildStats, build_market_event_dataset, save_dataset
from .massive_client import MassiveClient


NO_MARKET_DATA_PREFIX = "No grouped market data returned for "


def daterange(start: date, end: date):
    if end < start:
        raise ValueError("end must be on or after start")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _is_expected_closed_market_day(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and str(exc).startswith(NO_MARKET_DATA_PREFIX)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _client_stats(client: MassiveClient) -> dict[str, int]:
    stats = getattr(client, "stats", None)
    return stats.to_dict() if stats is not None else {}


def _stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in sorted(keys)}


def _checkpoint_paths(checkpoint_dir: Path, day: date) -> tuple[Path, Path]:
    stem = day.isoformat()
    return checkpoint_dir / f"{stem}.json", checkpoint_dir / f"{stem}.parquet"


def build_multi_day_dataset(
    client: MassiveClient,
    start: date,
    end: date,
    *,
    continue_on_error: bool = False,
    checkpoint_dir: Path | None = None,
    manifest_path: Path | None = None,
    **day_kwargs,
) -> tuple[pd.DataFrame, list[tuple[date, str]]]:
    """Build a date range with daily resumable checkpoints and a run manifest."""
    frames: list[pd.DataFrame] = []
    skipped: list[tuple[date, str]] = []
    manifest_days: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for day in daterange(start, end):
        status_path = data_path = None
        if checkpoint_dir is not None:
            status_path, data_path = _checkpoint_paths(checkpoint_dir, day)
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("state") == "complete":
                    kind = status.get("kind")
                    if kind == "events":
                        if data_path is None or not data_path.exists():
                            raise RuntimeError(f"checkpoint metadata exists but data is missing for {day}")
                        frames.append(pd.read_parquet(data_path))
                    elif kind == "closed_market":
                        skipped.append((day, str(status.get("reason", "closed_market"))))
                    elif kind != "no_events":
                        raise RuntimeError(f"unknown checkpoint kind for {day}: {kind}")
                    resumed = dict(status)
                    resumed["resumed"] = True
                    manifest_days.append(resumed)
                    continue

        # Avoid provider calls on weekends/holidays and use the same exchange
        # calendar later used to interpret regular sessions and early closes.
        if not is_us_equity_trading_day(day):
            reason = f"closed_market_calendar: {day.isoformat()}"
            skipped.append((day, reason))
            status = {
                "state": "complete",
                "kind": "closed_market",
                "trading_day": day.isoformat(),
                "reason": reason,
                "build_stats": DayBuildStats(trading_day=day.isoformat()).to_dict(),
                "client_stats_delta": {},
            }
            if status_path is not None:
                _atomic_json(status_path, status)
            manifest_days.append(status)
            continue

        before = _client_stats(client)
        stats = DayBuildStats(trading_day=day.isoformat())
        try:
            frame = build_market_event_dataset(client, day, build_stats=stats, **day_kwargs)
        except Exception as exc:
            if _is_expected_closed_market_day(exc):
                reason = f"closed_market: {exc}"
                skipped.append((day, reason))
                status = {
                    "state": "complete",
                    "kind": "closed_market",
                    "trading_day": day.isoformat(),
                    "reason": reason,
                    "build_stats": stats.to_dict(),
                    "client_stats_delta": _stats_delta(before, _client_stats(client)),
                }
                if status_path is not None:
                    _atomic_json(status_path, status)
                manifest_days.append(status)
                continue

            error_status = {
                "state": "error",
                "kind": "error",
                "trading_day": day.isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "build_stats": stats.to_dict(),
                "client_stats_delta": _stats_delta(before, _client_stats(client)),
            }
            manifest_days.append(error_status)
            if manifest_path is not None:
                _atomic_json(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "started_at_utc": started_at,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "days": manifest_days,
                        "client_stats": _client_stats(client),
                        "complete": False,
                    },
                )
            if not continue_on_error:
                raise
            skipped.append((day, f"ERROR {type(exc).__name__}: {exc}"))
            continue

        kind = "no_events" if frame.empty else "events"
        if not frame.empty:
            frames.append(frame)
            if data_path is not None:
                save_dataset(frame, data_path)

        status = {
            "state": "complete",
            "kind": kind,
            "trading_day": day.isoformat(),
            "build_stats": stats.to_dict(),
            "client_stats_delta": _stats_delta(before, _client_stats(client)),
        }
        if status_path is not None:
            _atomic_json(status_path, status)
        manifest_days.append(status)

    result = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["trading_day", "ticker", "timestamp_ms", "threshold_pct"])
        .reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )

    if manifest_path is not None:
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "days": manifest_days,
                "client_stats": _client_stats(client),
                "event_rows": int(len(result)),
                "event_tickers": int(result["ticker"].nunique()) if not result.empty else 0,
                "complete": True,
            },
        )

    return result, skipped


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m victory_trader.multi_day")
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-price", type=float, default=0.50)
    parser.add_argument("--max-price", type=float, default=20.0)
    parser.add_argument("--min-high-return", type=float, default=10.0)
    parser.add_argument("--min-dollar-volume", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--request-interval", type=float, default=12.5)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/run-manifest.json"))
    args = parser.parse_args()

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=args.request_interval,
    )
    frame, skipped = build_multi_day_dataset(
        client,
        args.start,
        args.end,
        min_price=args.min_price,
        max_price=args.max_price,
        min_high_return_pct=args.min_high_return,
        min_day_dollar_volume=args.min_dollar_volume,
        max_candidates=args.max_candidates,
        checkpoint_dir=args.checkpoint_dir,
        manifest_path=args.manifest,
    )

    output = args.output or Path("data/events") / f"market_events_{args.start.isoformat()}_{args.end.isoformat()}.parquet"
    if not frame.empty:
        save_dataset(frame, output)
        print(
            f"Saved {len(frame)} event rows across {frame['ticker'].nunique()} tickers "
            f"and {frame['trading_day'].nunique()} trading days to {output}"
        )
    else:
        print("No qualifying momentum events found in the requested range.")
    if skipped:
        print(f"Skipped {len(skipped)} closed/error calendar days.")
        for day, reason in skipped[:10]:
            print(f"  {day}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
