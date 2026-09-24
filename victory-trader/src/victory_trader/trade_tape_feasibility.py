"""Request 170: historical trade-tape feasibility probe.

No new market dates and no policy tuning. Deterministically sample already-opened
Request-166 evaluation POSITION states and query only trades with SIP timestamps
at or before each decision state.

This establishes whether tick-level trades are available and dense enough to be
a genuinely new causal POSITION information source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import load_settings
from .massive_client import MassiveClient

REQUEST_ID = 170
SAMPLE_PER_DAY = 10
LOOKBACK_MS = 60_000
RECENT_MS = 10_000

MIN_ANY_TRADE_COVERAGE = 0.80
MIN_FIVE_TRADE_COVERAGE = 0.70
MIN_RECENT_TRADE_COVERAGE = 0.60


def _sample_states(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    work = frame.sort_values(
        ["trading_day", "state_t", "ticker", "hot_t"],
        kind="stable",
    )
    for _, group in work.groupby(
        work["trading_day"].astype(str),
        sort=True,
    ):
        group = group.reset_index(drop=True)
        if len(group) > SAMPLE_PER_DAY:
            idx = np.linspace(
                0,
                len(group) - 1,
                SAMPLE_PER_DAY,
                dtype=int,
            )
            group = group.iloc[idx].copy()
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def _trade_features(rows: list[dict], state_t: int) -> dict[str, object]:
    records: list[tuple[int, float, float]] = []
    for row in rows:
        ts = pd.to_numeric(
            pd.Series([row.get("sip_timestamp")]), errors="coerce"
        ).iloc[0]
        price = pd.to_numeric(
            pd.Series([row.get("price")]), errors="coerce"
        ).iloc[0]
        size = pd.to_numeric(
            pd.Series([row.get("size")]), errors="coerce"
        ).iloc[0]
        if (
            pd.notna(ts)
            and pd.notna(price)
            and pd.notna(size)
            and int(ts) <= int(state_t) * 1_000_000
            and float(price) > 0
            and float(size) > 0
        ):
            records.append((int(ts), float(price), float(size)))
    records.sort(key=lambda item: item[0])

    if not records:
        return {
            "trade_count_60s": 0,
            "trade_volume_60s": 0.0,
            "trade_count_10s": 0,
            "trade_volume_10s": 0.0,
            "last_trade_age_ms": None,
            "trade_price_return_60s_pct": None,
            "tick_volume_imbalance": None,
            "median_intertrade_ms": None,
        }

    timestamps = np.asarray([x[0] for x in records], dtype=np.int64)
    prices = np.asarray([x[1] for x in records], dtype=float)
    sizes = np.asarray([x[2] for x in records], dtype=float)
    target_ns = int(state_t) * 1_000_000
    recent = timestamps >= (int(state_t) - RECENT_MS) * 1_000_000

    delta = np.diff(prices, prepend=prices[0])
    signs = np.sign(delta)
    last = 0.0
    for i in range(len(signs)):
        if signs[i] == 0:
            signs[i] = last
        else:
            last = signs[i]
    signed_volume = float(np.sum(signs * sizes))
    total_volume = float(np.sum(sizes))

    intertrade = np.diff(timestamps) / 1_000_000.0
    return {
        "trade_count_60s": int(len(records)),
        "trade_volume_60s": total_volume,
        "trade_count_10s": int(np.sum(recent)),
        "trade_volume_10s": float(np.sum(sizes[recent])),
        "last_trade_age_ms": float(
            (target_ns - int(timestamps[-1])) / 1_000_000.0
        ),
        "trade_price_return_60s_pct": (
            float((prices[-1] / prices[0] - 1.0) * 100.0)
            if prices[0] > 0
            else None
        ),
        "tick_volume_imbalance": (
            float(signed_volume / total_volume)
            if total_volume > 0
            else None
        ),
        "median_intertrade_ms": (
            float(np.median(intertrade))
            if len(intertrade)
            else None
        ),
    }


def run(
    evaluation_path: Path,
    output_path: Path,
    details_path: Path,
) -> int:
    frame = pd.read_parquet(evaluation_path)
    sample = _sample_states(frame)
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive-trade-tape-probe"),
        request_interval_seconds=0.05,
    )

    details: list[dict[str, object]] = []
    authorization_error = None
    for row in sample.to_dict("records"):
        state_t = int(row["state_t"])
        ticker = str(row["ticker"]).upper()
        try:
            trades = client.trades(
                ticker,
                timestamp_gte=(state_t - LOOKBACK_MS) * 1_000_000,
                timestamp_lte=state_t * 1_000_000,
                limit=50_000,
                order="asc",
            )
        except requests.HTTPError as exc:
            status = getattr(
                getattr(exc, "response", None),
                "status_code",
                None,
            )
            if status in {401, 403}:
                authorization_error = (
                    f"historical trade authorization failed HTTP {status}"
                )
                break
            raise

        record = {
            "trading_day": str(row["trading_day"]),
            "ticker": ticker,
            "hot_t": int(row["hot_t"]),
            "state_t": state_t,
        }
        record.update(_trade_features(trades, state_t))
        details.append(record)

    detail_frame = pd.DataFrame(details)
    sampled = int(len(sample))
    observed = int(len(detail_frame))
    if observed:
        any_trade = detail_frame["trade_count_60s"].gt(0)
        five_trade = detail_frame["trade_count_60s"].ge(5)
        recent_trade = detail_frame["trade_count_10s"].gt(0)
        any_coverage = float(any_trade.mean())
        five_coverage = float(five_trade.mean())
        recent_coverage = float(recent_trade.mean())
    else:
        any_coverage = five_coverage = recent_coverage = 0.0

    authorization_ok = authorization_error is None
    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "sampled_states": sampled,
        "queried_states_before_stop": observed,
        "authorization_error": authorization_error,
        "client_stats": client.stats.to_dict(),
        "any_trade_60s_coverage": any_coverage,
        "at_least_5_trades_60s_coverage": five_coverage,
        "any_trade_10s_coverage": recent_coverage,
        "gate": {
            "authorization_ok": authorization_ok,
            "any_trade_60s": (
                any_coverage >= MIN_ANY_TRADE_COVERAGE
            ),
            "five_trades_60s": (
                five_coverage >= MIN_FIVE_TRADE_COVERAGE
            ),
            "recent_trade_10s": (
                recent_coverage >= MIN_RECENT_TRADE_COVERAGE
            ),
        },
    }
    summary["feasibility_pass"] = bool(
        authorization_ok
        and observed == sampled
        and all(summary["gate"].values())
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    detail_frame.to_csv(details_path, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()
    return run(args.evaluation, args.output, args.details)


if __name__ == "__main__":
    raise SystemExit(main())
