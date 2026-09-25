from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.transition_recurrent_hold_exit import (
    attach_position_transitions,
)


def test_position_transitions_capture_recovery_and_deterioration() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 1.0,
                "entry_to_current_close_pct": 2.0,
                "drawdown_from_peak_pct": 0.0,
                "recovery_from_trough_pct": 2.0,
                "attention_score": 1.0,
                "attention_rank": 4.0,
                "log_minute_volume": 4.0,
                "log_minute_transactions": 3.0,
                "active_seconds_60": 40.0,
                "sec_last5_return_pct": 0.6,
                "sec_last10_return_pct": 1.0,
                "sec_close_vs_vwap_pct": 0.8,
                "sec_volume_burst_5": 1.1,
                "sec_transactions_burst_5": 1.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 2.0,
                "entry_to_current_close_pct": 0.5,
                "drawdown_from_peak_pct": -1.5,
                "recovery_from_trough_pct": 0.5,
                "attention_score": 0.8,
                "attention_rank": 6.0,
                "log_minute_volume": 3.7,
                "log_minute_transactions": 2.8,
                "active_seconds_60": 32.0,
                "sec_last5_return_pct": -0.4,
                "sec_last10_return_pct": 0.1,
                "sec_close_vs_vwap_pct": -0.2,
                "sec_volume_burst_5": 0.7,
                "sec_transactions_burst_5": 0.8,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 3.0,
                "entry_to_current_close_pct": 1.4,
                "drawdown_from_peak_pct": -0.6,
                "recovery_from_trough_pct": 1.4,
                "attention_score": 0.95,
                "attention_rank": 5.0,
                "log_minute_volume": 4.1,
                "log_minute_transactions": 3.1,
                "active_seconds_60": 44.0,
                "sec_last5_return_pct": 0.7,
                "sec_last10_return_pct": 0.5,
                "sec_close_vs_vwap_pct": 0.4,
                "sec_volume_burst_5": 1.4,
                "sec_transactions_burst_5": 1.3,
            },
        ]
    )
    out = attach_position_transitions(frame)
    row2 = out.loc[out["minutes_held"].eq(2.0)].iloc[0]
    row3 = out.loc[out["minutes_held"].eq(3.0)].iloc[0]
    assert row2["position_return_change_1m_pct"] == pytest.approx(-1.5)
    assert row3["position_return_change_1m_pct"] == pytest.approx(0.9)
    assert row3["position_return_accel_1m_pct"] == pytest.approx(2.4)
    assert row3["position_attention_change_1m"] > 0
    assert row3["position_drawdown_change_1m_pct"] > 0
