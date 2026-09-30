import pandas as pd

from victory_trader.local_turn_ablation import _policy_metrics


def test_policy_metrics_counts_days_beating_immediate():
    decisions = pd.DataFrame({
        "trading_day": ["2026-05-05", "2026-05-05", "2026-05-06"],
        "action": ["ENTER", "SKIP", "ENTER"],
        "realized_utility_pct": [3.0, 0.0, 1.0],
        "immediate_utility_pct": [1.0, 1.0, 2.0],
        "delay_s": [5.0, None, 0.0],
        "entry_price_improvement_pct": [0.5, None, 0.0],
        "chosen_mfe_fixed_pct": [12.0, 0.0, 4.0],
        "immediate_mfe_fixed_pct": [10.0, 0.0, 4.0],
    })
    result = _policy_metrics(decisions)
    assert result["episodes"] == 3
    assert result["entered"] == 2
    assert result["beat_immediate_days"] == 1
    assert result["plus10_rate"] == 0.5
