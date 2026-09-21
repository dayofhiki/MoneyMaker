from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.market_relative_attention_probe import (
    MARKET_RELATIVE_FEATURES,
    enrich_market_relative_features,
    fit_and_evaluate,
)
from victory_trader.second_path_attention_probe import BASELINE_FEATURES


def _scan_frame() -> pd.DataFrame:
    rows = []
    tickers = ["AAA", "BBB", "CCC"]
    for minute in range(30):
        for index, ticker in enumerate(tickers):
            base = 10.0 + index
            drift = (index + 1) * 0.002 * minute
            close = base * (1.0 + drift)
            open_price = close / (1.0 + 0.001 * (index + 1))
            rows.append(
                {
                    "trading_day": "2026-01-20",
                    "ticker": ticker,
                    "t": (minute + 1) * 60_000,
                    "bar_start_t": minute * 60_000,
                    "o": open_price,
                    "h": close * 1.002,
                    "l": open_price * 0.998,
                    "c": close,
                    "v": 1000.0 * (1 + index) * (1 + minute / 10),
                    "n": 100.0 * (1 + index) * (1 + minute / 20),
                    "previous_close": base,
                    "return_from_previous_close_pct": (close / base - 1.0) * 100.0,
                    "attention_score": (index + 1) / len(tickers),
                    "runner_cross_now": False,
                }
            )
    return pd.DataFrame(rows)


def test_market_relative_features_are_future_invariant():
    clean = _scan_frame()
    poisoned = clean.copy()
    final_t = int(poisoned["t"].max())
    future = poisoned["t"].eq(final_t)
    poisoned.loc[future, ["c", "h", "v", "n"]] = [999.0, 1000.0, 9e9, 9e8]

    clean_features = enrich_market_relative_features(clean)
    poisoned_features = enrich_market_relative_features(poisoned)
    before_final = clean_features["t"].lt(final_t)

    pd.testing.assert_frame_equal(
        clean_features.loc[
            before_final,
            ["trading_day", "ticker", "t", *MARKET_RELATIVE_FEATURES],
        ].reset_index(drop=True),
        poisoned_features.loc[
            before_final,
            ["trading_day", "ticker", "t", *MARKET_RELATIVE_FEATURES],
        ].reset_index(drop=True),
    )


def test_market_relative_return_rank_orders_same_timestamp_cross_section():
    frame = enrich_market_relative_features(_scan_frame())
    latest = frame.loc[frame["t"].eq(frame["t"].max())].set_index("ticker")

    assert latest.loc["CCC", "market_rank_return_1m"] > latest.loc[
        "BBB", "market_rank_return_1m"
    ]
    assert latest.loc["BBB", "market_rank_return_1m"] > latest.loc[
        "AAA", "market_rank_return_1m"
    ]


def test_fit_and_evaluate_uses_frozen_later_january_sessions():
    rng = np.random.default_rng(23)
    days = [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
        "2026-01-16",
        "2026-01-20",
        "2026-01-21",
        "2026-01-22",
        "2026-01-23",
        "2026-01-26",
    ]
    rows = []
    for day_index, day in enumerate(days):
        for i in range(20):
            base_signal = i / 20.0
            relative_signal = ((i * 3 + day_index) % 20) / 20.0
            row = {
                "trading_day": day,
                "ticker": f"T{i:03d}",
                "remaining_episode_peak_return_pct": (
                    base_signal + relative_signal + rng.normal(0, 0.01)
                ),
                "runner_in_remaining_episode": bool(relative_signal > 0.7),
                "episode_duration_minutes": 3.0 + relative_signal,
            }
            for feature in BASELINE_FEATURES:
                row[feature] = base_signal
            for feature in MARKET_RELATIVE_FEATURES:
                row[feature] = relative_signal
            rows.append(row)

    modeled, summary = fit_and_evaluate(pd.DataFrame(rows))

    assert summary["eval_days"] == days[-5:]
    assert summary["fit_days"] == days[:-5]
    assert summary["fit_rows"] == 220
    assert summary["eval_rows"] == 100
    assert set(
        modeled.loc[modeled["split"].eq("eval"), "trading_day"]
    ) == set(days[-5:])
    assert modeled["baseline_prediction"].notna().all()
    assert modeled["extended_prediction"].notna().all()
