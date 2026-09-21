from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.attention_replay import (
    build_market_scan_frame,
    replay_attention,
    summarize_attention_replay,
)
from victory_trader.attention_runtime import AttentionConfig

MINUTE_MS = 60_000


def _bars() -> pd.DataFrame:
    rows = []
    closes = {
        "AAA": [10.0, 10.4, 10.8, 11.1],
        "BBB": [10.0, 10.1, 10.2, 10.3],
        "CCC": [10.0, 9.9, 9.8, 9.7],
    }
    for minute in range(4):
        for ticker, values in closes.items():
            rows.append(
                {
                    "trading_day": "2026-01-02",
                    "ticker": ticker,
                    "t": minute * MINUTE_MS,
                    "c": values[minute],
                    "v": 1_000 + minute,
                }
            )
    return pd.DataFrame(rows)


def _prior() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trading_day": ["2026-01-02"] * 4,
            "ticker": ["AAA", "BBB", "CCC", "EXPENSIVE"],
            "previous_close": [10.0, 10.0, 10.0, 25.0],
            "eligible": [True, True, True, True],
        }
    )


def _config() -> AttentionConfig:
    return AttentionConfig(
        watch_enter_score=0.50,
        watch_exit_score=0.30,
        hot_enter_score=0.80,
        hot_exit_score=0.60,
        max_watch=1,
        max_hot=1,
        drop_after_missed_batches=2,
    )


def test_market_scan_score_uses_only_same_timestamp_cross_section():
    frame = build_market_scan_frame(_bars(), _prior())

    last = frame.loc[frame["t"].eq(4 * MINUTE_MS)].set_index("ticker")
    assert last.loc["AAA", "attention_score"] == 1.0
    assert last.loc["BBB", "attention_score"] == pytest.approx(2 / 3)
    assert last.loc["CCC", "attention_score"] == pytest.approx(1 / 3)
    assert last.loc["AAA", "runner_cross_now"]
    assert frame.loc[
        frame["ticker"].eq("AAA"), "runner_cross_now"
    ].sum() == 1


def test_market_scan_filters_ineligible_and_out_of_price_universe():
    bars = pd.concat(
        [
            _bars(),
            pd.DataFrame(
                [
                    {
                        "trading_day": "2026-01-02",
                        "ticker": "EXPENSIVE",
                        "t": 0,
                        "c": 26.0,
                        "v": 100,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    prior = _prior()
    prior.loc[prior["ticker"].eq("CCC"), "eligible"] = False

    frame = build_market_scan_frame(bars, prior)

    assert set(frame["ticker"]) == {"AAA", "BBB"}


def test_replay_is_chronological_and_captures_runner_by_cross():
    scan = build_market_scan_frame(_bars(), _prior())
    trace = replay_attention(scan, _config())
    summary = summarize_attention_replay(trace, scan)

    aaa = trace.loc[trace["ticker"].eq("AAA")]
    assert aaa["t"].is_monotonic_increasing
    assert aaa.iloc[-1]["state"] == "hot"
    assert summary["runner_episodes"] == 1
    assert summary["runner_watch_or_better_by_cross"] == 1
    assert summary["runner_hot_or_position_by_cross"] == 1
    assert summary["runner_watch_or_better_capture_rate"] == 1.0
    assert summary["runner_hot_or_position_capture_rate"] == 1.0


def test_replay_reports_capacity_and_resolution_demand():
    scan = build_market_scan_frame(_bars(), _prior())
    trace = replay_attention(scan, _config())
    summary = summarize_attention_replay(trace, scan)

    assert summary["max_watch_occupancy"] <= 1
    assert summary["max_hot_occupancy"] <= 1
    assert summary["grouped_minute_requests"] > 0
    assert summary["minute_bar_requests"] > 0
    assert summary["second_bar_requests"] > 0
    assert summary["trade_nbbo_requests"] == 0
    assert summary["state_changes"] > 0


def test_future_outcome_columns_cannot_change_attention_trace():
    scan = build_market_scan_frame(_bars(), _prior())
    poisoned = scan.copy()
    poisoned["buy_return_30m_base_net_return_pct"] = [
        1_000_000.0 if i % 2 else -1_000_000.0 for i in range(len(poisoned))
    ]

    clean_trace = replay_attention(scan, _config())
    poisoned_trace = replay_attention(poisoned, _config())

    pd.testing.assert_frame_equal(clean_trace, poisoned_trace)


def test_duplicate_ticker_timestamp_is_rejected():
    scan = build_market_scan_frame(_bars(), _prior())
    duplicate = pd.concat([scan, scan.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate ticker/timestamp"):
        replay_attention(duplicate, _config())


def test_sessions_are_replayed_with_fresh_attention_state():
    first = build_market_scan_frame(_bars(), _prior())
    second_bars = _bars().assign(
        trading_day="2026-01-05",
        t=lambda frame: frame["t"] + 3 * 24 * 60 * MINUTE_MS,
    )
    second_prior = _prior().assign(trading_day="2026-01-05")
    second = build_market_scan_frame(second_bars, second_prior)

    trace = replay_attention(pd.concat([first, second]), _config())
    starts = trace.sort_values("t").groupby("trading_day").first()

    assert set(starts.index) == {"2026-01-02", "2026-01-05"}
    assert starts.loc["2026-01-02", "t"] < starts.loc["2026-01-05", "t"]


def test_market_scan_decision_time_is_bar_completion_time():
    frame = build_market_scan_frame(_bars(), _prior())

    first = frame.sort_values(["t", "ticker"]).iloc[0]
    assert first["bar_start_t"] == 0
    assert first["t"] == MINUTE_MS
    assert (frame["t"] - frame["bar_start_t"]).eq(MINUTE_MS).all()
