from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from victory_trader.expanded_adaptive_horizon_enrichment import HORIZONS
from victory_trader.expanded_opportunity_shared_q import (
    PRIMARY,
    SHARED,
    _action_features,
    apply_policy_correction,
    decision_feasible,
    opportunity_target,
    policy_level_correction,
    select_policies,
)


def _labels(row: dict[str, object], value: float = 1.0) -> None:
    for horizon in HORIZONS:
        row[f"buy_return_{horizon}m_pct"] = value + 1.0
        row[f"buy_return_{horizon}m_base_net_return_pct"] = value
        row[f"buy_return_{horizon}m_stress_net_return_pct"] = value - 1.0


def test_clock_feasibility_does_not_use_future_label_availability() -> None:
    frame = pd.DataFrame(
        {
            "minutes_from_regular_open": [359.0, 388.0, 389.0],
            "buy_return_30m_base_net_return_pct": [np.nan, 2.0, 2.0],
        }
    )
    assert decision_feasible(frame, 30).tolist() == [True, False, False]
    assert decision_feasible(frame, 1).tolist() == [True, True, False]


def test_opportunity_target_uses_only_clock_feasible_actions() -> None:
    row: dict[str, object] = {"minutes_from_regular_open": 388.0}
    for horizon in HORIZONS:
        row[f"buy_return_{horizon}m_base_net_return_pct"] = -1.0
    row["buy_return_30m_base_net_return_pct"] = 100.0
    row["buy_return_1m_base_net_return_pct"] = 0.5
    target, valid = opportunity_target(pd.DataFrame([row]))
    assert valid.iloc[0]
    assert target.iloc[0]

    row["buy_return_1m_base_net_return_pct"] = -0.5
    target, valid = opportunity_target(pd.DataFrame([row]))
    assert valid.iloc[0]
    assert not target.iloc[0]


def test_action_features_share_state_scale_and_encode_horizon(monkeypatch) -> None:
    frame = pd.DataFrame(
        [{"minutes_from_regular_open": 10.0, "return_from_previous_close_pct": 12.0}]
    )
    monkeypatch.setattr(
        "victory_trader.expanded_opportunity_shared_q._feature_frame",
        lambda value: value.copy(),
    )
    features = _action_features(frame, 5)
    assert features.iloc[0]["action_horizon_min"] == pytest.approx(5.0)
    assert features.iloc[0]["action_is_5m"] == pytest.approx(1.0)
    assert features.iloc[0]["action_is_15m"] == pytest.approx(0.0)


def test_policy_level_correction_is_computed_after_max_action_selection() -> None:
    rows: list[dict[str, object]] = []
    for day in range(1, 7):
        row: dict[str, object] = {
            "trading_day": f"2025-12-{day:02d}",
            "shared_best_horizon_min": 2,
            "shared_best_raw_q_pct": 2.0,
            "opportunity_probability": 0.8,
            "minutes_from_regular_open": 100.0,
        }
        _labels(row, value=0.5)
        rows.append(row)
    summary = policy_level_correction(pd.DataFrame(rows))
    assert summary["selected_days"] == 6
    assert summary["policy_level_correction_pct"] == pytest.approx(1.5)

    corrected = apply_policy_correction(pd.DataFrame(rows), summary)
    assert corrected.iloc[0]["shared_best_corrected_q_pct"] == pytest.approx(0.5)


def test_primary_requires_gate_but_shared_comparator_does_not() -> None:
    rows: list[dict[str, object]] = []
    for ticker, gate in (("AAA", 0.8), ("BBB", 0.2)):
        row: dict[str, object] = {
            "trading_day": "2026-01-02",
            "ticker": ticker,
            "t": 1767364200000,
            "entry_price": 5.0,
            "adaptive_15m_hurdle_ev_pct": -0.1,
            "adaptive_best_hurdle_ev_pct": -0.1,
            "adaptive_best_horizon_min": 30,
            "shared_best_horizon_min": 2,
            "shared_best_corrected_q_pct": 0.4,
            "opportunity_probability": gate,
        }
        _labels(row)
        rows.append(row)
    trades, attempts = select_policies(pd.DataFrame(rows))
    assert set(trades.loc[trades["policy"].eq(SHARED), "ticker"]) == {"AAA", "BBB"}
    assert set(trades.loc[trades["policy"].eq(PRIMARY), "ticker"]) == {"AAA"}
    assert set(attempts.loc[attempts["policy"].eq(PRIMARY), "ticker"]) == {"AAA"}
