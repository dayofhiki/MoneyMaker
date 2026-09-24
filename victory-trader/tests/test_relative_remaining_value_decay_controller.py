from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.relative_remaining_value_decay_controller import (
    MIN_COMPLETION_COVERAGE,
    MIN_START_COVERAGE,
    REQUEST_ID,
    add_trend_columns,
)


def test_request188_contract_is_frozen():
    assert REQUEST_ID == 188
    assert MIN_START_COVERAGE == 0.85
    assert MIN_COMPLETION_COVERAGE == 0.90


def test_request188_uses_within_episode_score_change():
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-06-23"] * 3,
            "ticker": ["TEST"] * 3,
            "hot_t": [1000] * 3,
            "minutes_held": [1, 2, 3],
            "consensus_hurdle_ev_pct": [0.4, 0.6, 0.2],
            "excess_remaining_option_value_pct": [1.0, 1.5, 0.7],
        }
    )
    result = add_trend_columns(frame)
    assert pd.isna(result.loc[0, "score_delta_pct"])
    assert result.loc[1, "score_delta_pct"] == pytest.approx(0.2)
    assert result.loc[2, "score_delta_pct"] == pytest.approx(-0.4)
    assert result.loc[1, "actual_excess_value_delta_pct"] == pytest.approx(0.5)
    assert result.loc[2, "actual_excess_value_delta_pct"] == pytest.approx(-0.8)
