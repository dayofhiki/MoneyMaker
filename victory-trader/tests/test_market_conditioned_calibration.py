from __future__ import annotations

import pandas as pd

from victory_trader.market_conditioned_calibration import (
    add_causal_market_context,
)


def test_market_context_uses_only_strictly_earlier_hot_times() -> None:
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-05-11"] * 3,
            "ticker": ["AAA", "BBB", "CCC"],
            "t": [100, 100, 200],
            "minutes_since_open": [1.0, 1.0, 2.0],
            "predicted_persistent_tradability_pct": [1.0, 2.0, 3.0],
            "tradability_positive_probability": [0.6, 0.7, 0.8],
            "return_from_previous_close_pct": [10.0, 20.0, 30.0],
            "attention_score": [1.0, 3.0, 5.0],
            "second_rerank_probability": [0.2, 0.4, 0.6],
            "minute_body_return_pct": [1.0, 2.0, 3.0],
            "sec_last10_return_pct": [0.1, 0.2, 0.3],
            "sec_realized_vol_pct": [0.5, 0.7, 0.9],
        }
    )
    out = add_causal_market_context(frame)
    assert pd.isna(out.loc[0, "prior_hot_mean_return_pct"])
    assert pd.isna(out.loc[1, "prior_hot_mean_return_pct"])
    assert out.loc[2, "prior_hot_mean_return_pct"] == 15.0
