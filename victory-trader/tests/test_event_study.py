from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from victory_trader.event_study import measure_event_outcome, run_event_study
from victory_trader.events import CrossingEvent


ET = ZoneInfo("America/New_York")


def market_ts(hour: int, minute: int) -> int:
    return int(datetime(2026, 9, 15, hour, minute, tzinfo=ET).timestamp() * 1000)


def sample_bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"t": 0, "o": 10.0, "h": 10.1, "l": 9.9, "c": 10.0, "v": 100},
            {"t": 60_000, "o": 10.0, "h": 11.2, "l": 10.0, "c": 11.0, "v": 200},
            {"t": 120_000, "o": 11.0, "h": 12.3, "l": 10.9, "c": 12.0, "v": 300},
            {"t": 180_000, "o": 12.0, "h": 12.8, "l": 11.7, "c": 12.5, "v": 350},
            {"t": 240_000, "o": 12.5, "h": 13.0, "l": 11.4, "c": 11.5, "v": 500},
            {"t": 300_000, "o": 11.5, "h": 11.8, "l": 11.0, "c": 11.2, "v": 400},
        ]
    )


def test_measure_event_outcome_uses_next_minute_open_entry():
    bars = sample_bars()
    event = CrossingEvent("TEST", 120_000, 20.0, 12.0, 10.0)
    outcome = measure_event_outcome(event, bars, horizons=(1, 2, 5))

    assert outcome.entry_timestamp_ms == 180_000
    assert outcome.entry_price == pytest.approx(12.0)
    assert outcome.future_returns_pct[1] == pytest.approx((12.5 / 12.0 - 1) * 100)
    assert outcome.future_returns_pct[2] == pytest.approx((11.5 / 12.0 - 1) * 100)
    assert outcome.future_returns_pct[5] is None
    assert outcome.signal_returns_pct[1] == pytest.approx((12.5 / 12.0 - 1) * 100)


def test_missing_next_minute_bar_means_trade_was_not_executable():
    bars = pd.DataFrame(
        [
            {"t": 0, "o": 10.0, "h": 10.1, "l": 9.9, "c": 10.0, "v": 100},
            {"t": 60_000, "o": 10.0, "h": 12.1, "l": 10.0, "c": 12.0, "v": 200},
            {"t": 180_000, "o": 12.0, "h": 13.2, "l": 11.8, "c": 13.0, "v": 300},
        ]
    )
    event = CrossingEvent("TEST", 60_000, 20.0, 12.0, 10.0)
    outcome = measure_event_outcome(event, bars, horizons=(1, 2))

    assert outcome.entry_price is None
    assert outcome.future_returns_pct == {1: None, 2: None}
    assert outcome.signal_returns_pct[1] is None
    assert outcome.signal_returns_pct[2] == pytest.approx((13.0 / 12.0 - 1) * 100)


def test_latency_sensitivity_uses_later_exact_open():
    event = CrossingEvent("TEST", 120_000, 20.0, 12.0, 10.0)
    delayed = measure_event_outcome(event, sample_bars(), horizons=(1,), entry_delay_minutes=1)
    assert delayed.entry_timestamp_ms == 240_000
    assert delayed.entry_price == pytest.approx(12.5)
    assert delayed.future_returns_pct[1] == pytest.approx((11.5 / 12.5 - 1) * 100)


def test_run_event_study_detects_first_crossing_per_threshold_and_latency_columns():
    result = run_event_study(
        ticker="test",
        bars=sample_bars(),
        previous_close=10.0,
        thresholds_pct=(10, 20),
        horizons=(1, 2),
    )

    assert list(result["threshold_pct"]) == [10.0, 20.0]
    assert list(result["timestamp_ms"]) == [60_000, 120_000]
    assert "return_1m_pct" in result.columns
    assert "signal_return_1m_pct" in result.columns
    assert "delay1_return_1m_pct" in result.columns
    assert "delay2_return_1m_base_net_return_pct" in result.columns
    assert set(result["entry_model"]) == {"next_minute_open"}


def test_regular_scope_ignores_premarket_only_crossing():
    bars = pd.DataFrame(
        [
            {"t": market_ts(8, 0), "o": 10.0, "h": 12.5, "l": 10.0, "c": 12.0, "v": 100},
            {"t": market_ts(9, 30), "o": 10.5, "h": 11.5, "l": 10.0, "c": 11.0, "v": 200},
            {"t": market_ts(9, 31), "o": 11.0, "h": 11.2, "l": 10.8, "c": 11.1, "v": 200},
        ]
    )
    result = run_event_study(
        "TEST",
        bars,
        previous_close=10.0,
        thresholds_pct=(20,),
        horizons=(1,),
        event_session_scope="regular",
        require_regular_entry=True,
    )
    assert result.empty


def test_last_regular_minute_signal_has_no_regular_entry():
    bars = pd.DataFrame(
        [
            {"t": market_ts(15, 58), "o": 10.0, "h": 10.5, "l": 9.9, "c": 10.2, "v": 100},
            {"t": market_ts(15, 59), "o": 10.2, "h": 12.2, "l": 10.2, "c": 12.0, "v": 500},
            {"t": market_ts(16, 0), "o": 12.1, "h": 12.5, "l": 12.0, "c": 12.4, "v": 100},
        ]
    )
    result = run_event_study(
        "TEST",
        bars,
        previous_close=10.0,
        thresholds_pct=(20,),
        horizons=(1,),
        event_session_scope="regular",
        require_regular_entry=True,
    )
    assert len(result) == 1
    assert result.iloc[0]["entry_price"] is None
    assert result.iloc[0]["tp2_sl1_status"] == "entry_unavailable"
