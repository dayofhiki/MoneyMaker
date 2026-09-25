from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.pullback_turn_transition_entry import attach_transition_features


def test_transition_features_capture_pullback_and_rebound() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": 1,
                "current_price": 10.0,
                "attention_score": 1.0,
                "attention_rank": 5.0,
                "log_minute_volume": 4.0,
                "log_minute_transactions": 3.0,
                "active_seconds_60": 40.0,
                "sec_last5_return_pct": 1.0,
                "sec_last10_return_pct": 2.0,
                "sec_close_vs_vwap_pct": 1.0,
                "sec_volume_burst_5": 1.0,
                "sec_transactions_burst_5": 1.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": 2,
                "current_price": 9.5,
                "attention_score": 0.8,
                "attention_rank": 7.0,
                "log_minute_volume": 3.8,
                "log_minute_transactions": 2.9,
                "active_seconds_60": 35.0,
                "sec_last5_return_pct": -0.5,
                "sec_last10_return_pct": 0.2,
                "sec_close_vs_vwap_pct": -0.2,
                "sec_volume_burst_5": 0.7,
                "sec_transactions_burst_5": 0.8,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": 3,
                "current_price": 9.8,
                "attention_score": 0.9,
                "attention_rank": 6.0,
                "log_minute_volume": 4.2,
                "log_minute_transactions": 3.2,
                "active_seconds_60": 45.0,
                "sec_last5_return_pct": 0.8,
                "sec_last10_return_pct": 0.6,
                "sec_close_vs_vwap_pct": 0.3,
                "sec_volume_burst_5": 1.3,
                "sec_transactions_burst_5": 1.2,
            },
        ]
    )
    out = attach_transition_features(frame)
    row2 = out.loc[out["minutes_since_hot"].eq(2)].iloc[0]
    row3 = out.loc[out["minutes_since_hot"].eq(3)].iloc[0]

    assert row2["watch_drawdown_from_peak_pct"] == pytest.approx(-5.0)
    assert row3["watch_rebound_from_trough_pct"] == pytest.approx(
        (9.8 / 9.5 - 1.0) * 100.0
    )
    assert row3["watch_price_change_1m_pct"] > 0
    assert row3["watch_attention_change_1m"] > 0
    assert row3["watch_log_volume_change_1m"] > 0
