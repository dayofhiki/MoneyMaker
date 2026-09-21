from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.expanded_adaptive_horizon_diagnostic import (
    HORIZONS,
    PRIMARY_POLICY,
    horizon_summary,
    path_diagnostics,
)


def _row(day: str, ticker: str, base: dict[int, float]) -> dict[str, object]:
    row: dict[str, object] = {
        "policy": PRIMARY_POLICY,
        "trading_day": day,
        "ticker": ticker,
        "month": day[:7],
    }
    for horizon in HORIZONS:
        b = base[horizon]
        row[f"buy_return_{horizon}m_base_net_return_pct"] = b
        row[f"buy_return_{horizon}m_pct"] = b + 1.0
        row[f"buy_return_{horizon}m_stress_net_return_pct"] = b - 1.0
    return row


def test_path_diagnostic_separates_horizon_and_candidate_misses() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "2026-01-02",
                "AAA",
                {1: 1.0, 2: 0.5, 5: -0.2, 10: -0.5, 15: -1.0, 30: -2.0},
            ),
            _row(
                "2026-01-03",
                "BBB",
                {1: -0.1, 2: -0.2, 5: -0.3, 10: -0.4, 15: -0.5, 30: -0.6},
            ),
            _row(
                "2026-01-04",
                "CCC",
                {1: -0.2, 2: 0.1, 5: 0.1, 10: 0.1, 15: 0.2, 30: 0.0},
            ),
        ]
    )
    result = path_diagnostics(frame).iloc[0]
    assert result["trades_with_15m"] == 3
    assert result["fixed15_positive_rate"] == pytest.approx(1 / 3)
    assert result["horizon_miss_rate"] == pytest.approx(1 / 3)
    assert result["candidate_miss_rate"] == pytest.approx(1 / 3)
    assert result["horizon_miss_rate_among_15m_nonpositive"] == pytest.approx(0.5)
    assert result["oracle_base_lift_vs_15m_pct"] > 0


def test_horizon_summary_preserves_cost_decomposition() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "2026-01-02",
                "AAA",
                {1: 1.0, 2: 0.5, 5: 0.0, 10: -0.5, 15: -1.0, 30: -2.0},
            ),
            _row(
                "2026-01-03",
                "BBB",
                {1: 0.0, 2: -0.5, 5: -1.0, 10: -1.5, 15: -2.0, 30: -3.0},
            ),
        ]
    )
    summary = horizon_summary(frame)
    one = summary.loc[summary["horizon_min"].eq(1)].iloc[0]
    assert one["evaluated"] == 2
    assert one["gross_mean_pct"] == pytest.approx(1.5)
    assert one["base_mean_pct"] == pytest.approx(0.5)
    assert one["modeled_base_cost_drag_mean_pct"] == pytest.approx(1.0)
