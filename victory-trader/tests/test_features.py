from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from victory_trader.event_study import run_event_study
from victory_trader.events import CrossingEvent
from victory_trader.features import extract_event_features


ET = ZoneInfo("America/New_York")


def ts(day: int, hour: int, minute: int) -> int:
    dt = datetime(2026, 9, day, hour, minute, tzinfo=ET)
    return int(dt.timestamp() * 1000)


def bar(t, price, volume, high=None, low=None, vw=None):
    return {
        "t": t,
        "o": price,
        "h": high if high is not None else price,
        "l": low if low is not None else price,
        "c": price,
        "v": volume,
        "vw": vw if vw is not None else price,
    }


def test_event_features_do_not_use_future_bars():
    bars = pd.DataFrame(
        [
            bar(ts(15, 9, 30), 10.0, 100),
            bar(ts(15, 9, 31), 11.0, 200),
            bar(ts(15, 9, 32), 12.0, 300, high=12.2),
            bar(ts(15, 9, 33), 30.0, 99_999_999, high=40.0, low=1.0),
        ]
    )
    event = CrossingEvent("TEST", ts(15, 9, 32), 20.0, 12.0, 10.0)

    with_future = extract_event_features(event, bars)
    without_future = extract_event_features(event, bars.iloc[:3])

    assert with_future == without_future
    assert with_future.cumulative_volume == 600.0
    assert with_future.hod_distance_pct == pytest.approx((12.0 / 12.2 - 1.0) * 100.0)


def test_historical_rvol_uses_prior_dates_at_same_clock_time():
    history = pd.DataFrame(
        [
            bar(ts(13, 9, 30), 10.0, 100),
            bar(ts(13, 9, 31), 10.0, 100),
            bar(ts(13, 9, 32), 10.0, 100),
            bar(ts(14, 9, 30), 10.0, 200),
            bar(ts(14, 9, 31), 10.0, 200),
            bar(ts(14, 9, 32), 10.0, 200),
        ]
    )
    target = pd.DataFrame(
        [
            bar(ts(15, 9, 30), 10.0, 300),
            bar(ts(15, 9, 31), 11.0, 300),
            bar(ts(15, 9, 32), 12.0, 300),
        ]
    )
    event = CrossingEvent("TEST", ts(15, 9, 32), 20.0, 12.0, 10.0)

    features = extract_event_features(event, target, history_bars=history)

    assert features.rvol_cumulative_20d == pytest.approx(2.0)
    assert features.rvol_5m_20d == pytest.approx(2.0)
    assert features.rvol_history_days == 2


def test_event_study_emits_model_features_and_future_outcomes_separately():
    bars = pd.DataFrame(
        [
            bar(ts(15, 9, 30), 10.0, 100),
            bar(ts(15, 9, 31), 11.0, 200),
            bar(ts(15, 9, 32), 12.0, 300),
            bar(ts(15, 9, 33), 12.6, 400),
        ]
    )

    result = run_event_study(
        "TEST",
        bars,
        previous_close=10.0,
        thresholds_pct=(20,),
        horizons=(1,),
    )

    assert result.iloc[0]["cumulative_volume"] == 600.0
    assert result.iloc[0]["return_1m_pct"] == pytest.approx(5.0)
    assert "trailing_return_5m_pct" in result.columns
    assert "return_5m_pct" not in result.columns
    assert "vwap_distance_pct" in result.columns
    assert "rvol_cumulative_20d" in result.columns
