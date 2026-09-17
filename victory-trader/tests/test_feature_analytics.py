import pandas as pd
import pytest

from victory_trader.feature_analytics import summarize_rvol_net


def test_rvol_summary_uses_after_cost_return_not_gross():
    frame = pd.DataFrame(
        {
            "threshold_pct": [20.0, 20.0],
            "rvol_cumulative_20d": [3.0, 3.5],
            "return_5m_pct": [5.0, 4.0],
            "return_5m_base_net_return_pct": [-1.0, -2.0],
        }
    )
    result = summarize_rvol_net(frame, horizon_min=5, scenario="base")
    row = result.iloc[0]
    assert row["rvol_bucket"] == "2-5x"
    assert row["mean_return_pct"] == pytest.approx(-1.5)
    assert row["positive_rate"] == 0.0
    assert row["scenario"] == "base"
