from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from victory_trader.event_time_executable_entry import (
    attach_event_time_execution,
    decide_event_time,
)


def _sources() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
            }
        ]
    )


def _scored(minutes, evs, advantages=None) -> pd.DataFrame:
    advantages = advantages or [1.0] * len(minutes)
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": minute,
                "predicted_entry_ev_pct": ev,
                "predicted_relative_advantage_pct": advantage,
            }
            for minute, ev, advantage in zip(
                minutes, evs, advantages, strict=True
            )
        ]
    )


def test_event_time_decision_waits_through_silent_minute() -> None:
    frame = _scored([1, 3, 4, 5], [-1.0, 1.0, 1.0, 1.0])
    decision = decide_event_time(frame, _sources()).iloc[0]
    assert decision.action == "ENTER"
    assert decision.entry_minute_after_hot == 3
    assert decision.stale_wait_actions == 1
    assert decision.observed_wait_actions == 1
    assert decision.wait_actions == 2


def test_event_time_decision_never_enters_on_stale_clock() -> None:
    frame = _scored([1, 4], [-1.0, -1.0])
    decision = decide_event_time(frame, _sources()).iloc[0]
    assert decision.action == "ABSTAIN"
    assert pd.isna(decision.entry_minute_after_hot)
    assert decision.stale_wait_actions == 3


def test_prediction_miss_remains_explicit() -> None:
    frame = _scored([1], [np.nan])
    decision = decide_event_time(frame, _sources()).iloc[0]
    assert decision.action == "PREDICTION_MISS"


def test_event_time_exit_uses_first_available_after_target() -> None:
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": 1,
                "enter_3m_base_pct": np.nan,
            }
        ]
    )
    positions = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 1.0,
                "exit_reference_open": 10.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 5.5,
                "exit_reference_open": 11.0,
            },
        ]
    )
    out = attach_event_time_execution(states, positions).iloc[0]
    assert pd.notna(out.enter_3m_event_base_pct)
    assert out.event_exit_minute_after_hot == pytest.approx(5.5)
    assert out.event_exit_delay_after_3m_min == pytest.approx(1.5)
    assert out.gross_return_pct == pytest.approx(10.0)


def test_event_time_exit_does_not_use_pre_target_price() -> None:
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": 1,
                "enter_3m_base_pct": np.nan,
            }
        ]
    )
    positions = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 1.0,
                "exit_reference_open": 10.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 3.9,
                "exit_reference_open": 100.0,
            },
        ]
    )
    out = attach_event_time_execution(states, positions).iloc[0]
    assert pd.isna(out.enter_3m_event_base_pct)
    assert pd.isna(out.event_exit_minute_after_hot)
