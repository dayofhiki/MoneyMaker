"""Massive Flat Files adapter for chronological market attention replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from math import isclose
from pathlib import Path

import pandas as pd

from .attention_replay import (
    build_market_scan_frame,
    replay_attention,
    summarize_attention_replay,
)
from .attention_runtime import AttentionConfig
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import (
    is_us_equity_trading_day,
    previous_us_equity_trading_day,
    regular_session_bounds,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .universe import (
    fetch_research_universe_metadata,
    split_tickers_from_payload,
)

FLATFILE_CACHE_DIR = Path("data/cache/massive-flatfiles")
REST_CACHE_DIR = Path("data/cache/massive")
MINUTE_VALUE_COLUMNS = ("o", "h", "l", "c", "v")


def _prepare_prior_closes(
    previous: pd.DataFrame,
    eligible: set[str],
    trading_day: date,
    previous_day: date,
    rest_client: MassiveClient,
) -> tuple[pd.DataFrame, int, int]:
    """Normalize duplicate closes, using exact-date REST bars for conflicts."""

    diagnostic_columns = [
        column
        for column in ("window_start", "volume", "transactions")
        if column in previous.columns
    ]
    prior = previous.loc[:, ["ticker", "close", *diagnostic_columns]].copy()
    prior["ticker"] = prior["ticker"].astype(str).str.strip().str.upper()
    prior["previous_close"] = pd.to_numeric(prior.pop("close"), errors="coerce")
    prior = prior.loc[prior["ticker"].isin(eligible)].copy()
    before = len(prior)

    duplicate = prior.duplicated("ticker", keep=False)
    if duplicate.any():
        duplicate_values = prior.loc[duplicate]
        distinct_counts = duplicate_values.groupby("ticker", sort=True)[
            "previous_close"
        ].nunique(dropna=False)
        conflicting = distinct_counts.loc[distinct_counts.gt(1)].index.tolist()
    else:
        conflicting = []

    resolved = 0
    for ticker in conflicting:
        payload = rest_client.daily_bars(
            ticker,
            previous_day,
            previous_day,
            adjusted=False,
        )
        results = list(payload.get("results") or [])
        rest_closes = [
            float(item["c"])
            for item in results
            if item.get("c") is not None
        ]
        candidates = prior.loc[
            prior["ticker"].eq(ticker), "previous_close"
        ].drop_duplicates().tolist()
        matching = [
            candidate
            for candidate in candidates
            if any(
                isclose(candidate, rest_close, rel_tol=1e-12, abs_tol=1e-12)
                for rest_close in rest_closes
            )
        ]
        if len(results) != 1 or len(matching) != 1:
            details = (
                prior.loc[
                    prior["ticker"].eq(ticker),
                    ["ticker", "previous_close", *diagnostic_columns],
                ]
                .sort_values(["ticker", *diagnostic_columns], kind="stable")
                .to_dict("records")
            )
            raise ValueError(
                "could not resolve conflicting prior-day Flat File closes "
                f"from exact-date REST bar: rows={details}, "
                f"rest_closes={rest_closes}"
            )
        selected = matching[0]
        remove = prior["ticker"].eq(ticker) & prior["previous_close"].ne(selected)
        prior = prior.loc[~remove].copy()
        resolved += 1

    prior = prior.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    prior = prior.loc[:, ["ticker", "previous_close"]]
    prior["trading_day"] = trading_day.isoformat()
    prior["eligible"] = True
    return prior, before - len(prior), resolved


def _rows_match(left: pd.Series, right: pd.Series) -> bool:
    for column in MINUTE_VALUE_COLUMNS:
        left_value = pd.to_numeric(pd.Series([left[column]]), errors="coerce").iloc[0]
        right_value = pd.to_numeric(pd.Series([right[column]]), errors="coerce").iloc[0]
        if pd.isna(left_value) or pd.isna(right_value):
            return False
        if column in {"o", "h", "l", "c"}:
            # Flat File and REST aggregates occasionally differ only by
            # provider serialization precision (for example 73.511306 versus
            # 73.511300). Accept sub-micro-relative price rounding while still
            # rejecting genuinely different duplicate price series.
            matches = isclose(
                float(left_value),
                float(right_value),
                rel_tol=1e-7,
                abs_tol=1e-6,
            )
        else:
            # Volume is integral in the source data and should match exactly
            # apart from numeric representation.
            matches = isclose(
                float(left_value),
                float(right_value),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        if not matches:
            return False
    return True


def _prepare_minute_bars(
    minutes: pd.DataFrame,
    eligible: set[str],
    day: date,
    rest_client: MassiveClient,
    open_ms: int,
    close_ms: int,
) -> tuple[pd.DataFrame, int, int]:
    """Normalize minute rows and resolve conflicting ticker series via REST."""

    frame = minutes.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["t"] = pd.to_numeric(frame["t"], errors="coerce")
    frame = frame.loc[
        frame["ticker"].isin(eligible)
        & frame["t"].between(open_ms, close_ms - 1, inclusive="both")
    ].copy()
    frame["t"] = frame["t"].astype("int64")
    before = len(frame)
    keys = ["ticker", "t"]
    duplicate_rows_collapsed = before - len(frame.drop_duplicates(keys))
    duplicate = frame.duplicated(keys, keep=False)
    conflicting_tickers: list[str] = []
    conflicting_keys: list[tuple[str, int]] = []
    if duplicate.any():
        duplicate_rows = frame.loc[duplicate, [*keys, *MINUTE_VALUE_COLUMNS]]
        distinct = duplicate_rows.groupby(keys, sort=True)[
            list(MINUTE_VALUE_COLUMNS)
        ].nunique(dropna=False)
        conflicting_keys = [
            (str(ticker), int(timestamp))
            for ticker, timestamp in distinct.index[distinct.gt(1).any(axis=1)]
        ]
        conflicting_tickers = sorted({str(key[0]) for key in conflicting_keys})

    resolved = 0
    for ticker in conflicting_tickers:
        payload = rest_client.minute_bars(ticker, day, adjusted=False)
        rest = pd.DataFrame(payload.get("results") or [])
        required = {"t", *MINUTE_VALUE_COLUMNS}
        if rest.empty or not required.issubset(rest.columns):
            raise ValueError(
                f"could not resolve conflicting minute bars for {ticker}: "
                "exact-date REST bars are empty or incomplete"
            )
        rest["t"] = pd.to_numeric(rest["t"], errors="coerce")
        rest = rest.loc[
            rest["t"].between(open_ms, close_ms - 1, inclusive="both")
        ].copy()
        rest["t"] = rest["t"].astype("int64")
        if rest.empty or rest.duplicated("t").any():
            raise ValueError(
                f"could not resolve conflicting minute bars for {ticker}: "
                "exact-date REST timestamps are empty or duplicated"
            )

        conflict_times = {
            timestamp
            for conflict_ticker, timestamp in conflicting_keys
            if conflict_ticker == ticker
        }
        ticker_duplicates = frame.loc[
            frame["ticker"].eq(ticker) & frame["t"].isin(conflict_times)
        ]
        rest_by_time = rest.set_index("t", drop=False)
        for timestamp, candidates in ticker_duplicates.groupby("t", sort=True):
            if timestamp not in rest_by_time.index:
                raise ValueError(
                    f"could not resolve conflicting minute bars for {ticker}: "
                    f"REST bar missing at {timestamp}"
                )
            reference = rest_by_time.loc[timestamp]
            matches = sum(
                _rows_match(candidate, reference)
                for _, candidate in candidates.iterrows()
            )
            if matches != 1:
                raise ValueError(
                    f"could not resolve conflicting minute bars for {ticker}: "
                    f"expected one Flat File match at {timestamp}, found {matches}"
                )

        replacement = rest.loc[:, ["t", *MINUTE_VALUE_COLUMNS]].copy()
        replacement["ticker"] = ticker
        replacement["n"] = rest["n"] if "n" in rest.columns else pd.NA
        frame = pd.concat(
            [frame.loc[~frame["ticker"].eq(ticker)], replacement],
            ignore_index=True,
        )
        resolved += 1

    frame = frame.drop_duplicates(keys, keep="first").reset_index(drop=True)
    frame["trading_day"] = day.isoformat()
    return frame, duplicate_rows_collapsed, resolved


def build_flatfile_scan_day(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    day: date,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Build one causal broad-scan day without completed-day selection."""

    bounds = regular_session_bounds(day)
    if bounds is None:
        return pd.DataFrame(), {
            "trading_day": day.isoformat(),
            "kind": "closed_market",
        }
    open_ms = int(bounds[0].timestamp() * 1000)
    close_ms = int(bounds[1].timestamp() * 1000)

    previous_day = previous_us_equity_trading_day(day)
    previous = store.day_aggregates(previous_day)
    minutes = store.minute_aggregates(day)
    metadata = fetch_research_universe_metadata(rest_client, day)
    split_tickers = split_tickers_from_payload(rest_client.splits_on(day))
    eligible = {
        ticker
        for ticker, item in metadata.items()
        if item.is_research_common_stock and ticker not in split_tickers
    }

    (
        prior,
        duplicate_prior_rows_collapsed,
        conflicting_prior_tickers_resolved,
    ) = _prepare_prior_closes(
        previous,
        eligible,
        day,
        previous_day,
        rest_client,
    )

    (
        minutes,
        duplicate_minute_rows_collapsed,
        conflicting_minute_tickers_resolved,
    ) = _prepare_minute_bars(
        minutes,
        eligible,
        day,
        rest_client,
        open_ms,
        close_ms,
    )

    scan = build_market_scan_frame(minutes, prior)
    return scan, {
        "trading_day": day.isoformat(),
        "kind": "scan",
        "previous_trading_day": previous_day.isoformat(),
        "point_in_time_common_stocks": len(metadata),
        "eligible_after_exchange_type_split": len(eligible),
        "same_day_split_exclusions": len(split_tickers),
        "duplicate_prior_rows_collapsed": duplicate_prior_rows_collapsed,
        "conflicting_prior_tickers_resolved": conflicting_prior_tickers_resolved,
        "duplicate_minute_rows_collapsed": duplicate_minute_rows_collapsed,
        "conflicting_minute_tickers_resolved": conflicting_minute_tickers_resolved,
        "regular_minute_input_rows": len(minutes),
        "scan_rows": len(scan),
        "scan_symbols": int(scan["ticker"].nunique()) if not scan.empty else 0,
        "runner_crossings": (
            int(scan["runner_cross_now"].sum()) if not scan.empty else 0
        ),
    }


def run_flatfile_attention_replay(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    start: date,
    end: date,
    *,
    config: AttentionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    scans: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    days: list[dict[str, int | str]] = []
    runtime_config = config or AttentionConfig()

    for day in daterange(start, end):
        if not is_us_equity_trading_day(day):
            days.append(
                {"trading_day": day.isoformat(), "kind": "closed_market"}
            )
            continue
        scan, day_summary = build_flatfile_scan_day(store, rest_client, day)
        days.append(day_summary)
        if scan.empty:
            continue
        scans.append(scan)
        traces.append(replay_attention(scan, runtime_config))

    scan_frame = pd.concat(scans, ignore_index=True) if scans else pd.DataFrame()
    trace_frame = (
        pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    )
    replay_summary = (
        summarize_attention_replay(trace_frame, scan_frame)
        if not trace_frame.empty
        else {"trace_rows": 0, "runner_episodes": 0}
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "attention_config": asdict(runtime_config),
        "score_definition": (
            "same-timestamp cross-sectional percentile of nominal return "
            "from prior close"
        ),
        "days": days,
        "replay": replay_summary,
        "flatfile_stats": store.stats.to_dict(),
    }
    return scan_frame, trace_frame, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.attention_flatfile_replay",
        description="Replay hierarchical attention on market-wide Massive minute files.",
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--scan-output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR,
    )
    rest_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=REST_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    scan, trace, summary = run_flatfile_attention_replay(
        store,
        rest_client,
        args.start,
        args.end,
    )
    for path in (args.scan_output, args.trace_output, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(args.scan_output, index=False, compression="zstd")
    trace.to_parquet(args.trace_output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
