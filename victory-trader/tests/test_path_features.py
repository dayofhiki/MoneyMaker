from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from victory_trader.events import CrossingEvent
from victory_trader.features import extract_event_features


ET = ZoneInfo("America/New_York")


def ts(hour: int, minute: int) -> int:
    dt = datetime(2026, 3, 10, hour, minute, tzinfo=ET)
    return int(dt.timestamp() * 1000)


def bar(hour: int, minute: int, close: float, high: float | None = None, low: float | None = None):
    return {
        "t": ts(hour, minute),
        "o": close,
        "h": close if high is None else high,
        "l": close if low is None else low,
        "c": close,
        "v": 1_000.0,
        "vw": close,
    }


def test_crossing_speed_uses_only_observed_closes():
    bars = pd.DataFrame(
        [
            bar(9, 30, 10.0),
            bar(9, 31, 11.0),
            bar(9, 32, 11.4),
            bar(9, 33, 12.1),
            bar(9, 34, 50.0),
        ]
    )
    event = CrossingEvent("TEST", ts(9, 33), 20.0, 12.1, 10.0)

    features = extract_event_features(event, bars)

    assert features.minutes_since_10pct_cross == pytest.approx(2.0)
    assert features.minutes_since_prior_threshold_cross == pytest.approx(2.0)


def test_prior_15m_path_excludes_event_bar_and_future_bars():
    bars = pd.DataFrame(
        [
            bar(9, 30, 10.0, high=10.2, low=9.8),
            bar(9, 31, 11.0, high=11.5, low=10.5),
            bar(9, 32, 12.0, high=20.0, low=1.0),
            bar(9, 33, 30.0, high=40.0, low=0.5),
        ]
    )
    event = CrossingEvent("TEST", ts(9, 32), 20.0, 12.0, 10.0)

    features = extract_event_features(event, bars)

    assert features.prior_15m_high_distance_pct == pytest.approx((12.0 / 11.5 - 1.0) * 100.0)
    assert features.prior_15m_low_rebound_pct == pytest.approx((12.0 / 9.8 - 1.0) * 100.0)
