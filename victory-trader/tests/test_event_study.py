import pandas as pd
import pytest

from victory_trader.event_study import measure_event_outcome, run_event_study
from victory_trader.events import CrossingEvent


def sample_bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"t": 0, "o": 10.0, "h": 10.1, "l": 9.9, "c": 10.0, "v": 100},
            {"t": 60_000, "o": 10.0, "h": 11.2, "l": 10.0, "c": 11.0, "v": 200},
            {"t": 120_000, "o": 11.0, "h": 12.3, "l": 10.9, "c": 12.0, "v": 300},
            {"t": 180_000, "o": 12.0, "h": 12.8, "l": 11.7, "c": 12.5, "v": 350},
            {"t": 240_000, "o": 12.5, "h": 13.0, "l": 11.4, "c": 11.5, "v": 500},
        ]
    )


def test_measure_event_outcome_uses_future_bars_only():
    bars = sample_bars()
    event = CrossingEvent(
        ticker="TEST",
        timestamp_ms=120_000,
        threshold_pct=20.0,
        price=12.0,
        previous_close=10.0,
    )

    outcome = measure_event_outcome(event, bars, horizons=(1, 2, 5))

    assert outcome.future_returns_pct[1] == pytest.approx((12.5 / 12.0 - 1) * 100)
    assert outcome.future_returns_pct[2] == pytest.approx((11.5 / 12.0 - 1) * 100)
    assert outcome.future_returns_pct[5] is None
    assert outcome.mfe_pct == pytest.approx((13.0 / 12.0 - 1) * 100)
    assert outcome.mae_pct == pytest.approx((11.4 / 12.0 - 1) * 100)


def test_run_event_study_detects_first_crossing_per_threshold():
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
    assert "mfe_pct" in result.columns
    assert "mae_pct" in result.columns
