from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from victory_trader.expanded_adaptive_horizon_enrichment import (
    HORIZONS,
    attach_labels,
    return_column,
)
from victory_trader.expanded_adaptive_hurdle_ev import (
    PRIMARY,
    select_policies,
)


def test_attach_labels_preserves_existing_15m_and_adds_other_horizons() -> None:
    anchors = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "t": 1767364200000,
                "buy_return_15m_pct": 2.0,
                "buy_return_15m_base_net_return_pct": 1.0,
                "buy_return_15m_stress_net_return_pct": 0.0,
            }
        ]
    )
    state = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": 1767364200000,
    }
    for horizon in HORIZONS:
        state[return_column(horizon, "gross")] = float(horizon) / 10.0
        state[return_column(horizon, "base")] = float(horizon) / 10.0 - 0.5
        state[return_column(horizon, "stress")] = float(horizon) / 10.0 - 1.0
    state["buy_return_15m_pct"] = 2.0
    state["buy_return_15m_base_net_return_pct"] = 1.0
    state["buy_return_15m_stress_net_return_pct"] = 0.0

    enriched = attach_labels(anchors, pd.DataFrame([state]))
    assert enriched.iloc[0]["buy_return_1m_pct"] == pytest.approx(0.1)
    assert enriched.iloc[0]["buy_return_30m_base_net_return_pct"] == pytest.approx(2.5)
    assert enriched.iloc[0]["buy_return_15m_base_net_return_pct"] == pytest.approx(1.0)


def test_adaptive_policy_chooses_highest_positive_hurdle_ev() -> None:
    row: dict[str, object] = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": 1767364200000,
        "entry_price": 5.0,
        "adaptive_best_horizon_min": 5,
        "adaptive_best_hurdle_ev_pct": 1.2,
        "adaptive_15m_hurdle_ev_pct": -0.1,
    }
    for horizon in HORIZONS:
        row[f"buy_return_{horizon}m_pct"] = 2.0
        row[f"buy_return_{horizon}m_base_net_return_pct"] = 1.0
        row[f"buy_return_{horizon}m_stress_net_return_pct"] = 0.0
    trades, attempts = select_policies(pd.DataFrame([row]))
    primary = trades.loc[trades["policy"].eq(PRIMARY)].iloc[0]
    assert int(primary["action_horizon_min"]) == 5
    assert primary["realized_base_net_return_pct"] == pytest.approx(1.0)
    assert attempts.loc[attempts["policy"].eq(PRIMARY), "evaluation_reason"].iloc[0] == "evaluated"


def test_adaptive_policy_abstains_when_all_action_value_is_nonpositive() -> None:
    row: dict[str, object] = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": 1767364200000,
        "entry_price": 5.0,
        "adaptive_best_horizon_min": 2,
        "adaptive_best_hurdle_ev_pct": -0.01,
        "adaptive_15m_hurdle_ev_pct": -0.2,
    }
    for horizon in HORIZONS:
        row[f"buy_return_{horizon}m_pct"] = 0.0
        row[f"buy_return_{horizon}m_base_net_return_pct"] = -1.0
        row[f"buy_return_{horizon}m_stress_net_return_pct"] = -2.0
    trades, attempts = select_policies(pd.DataFrame([row]))
    assert not trades["policy"].eq(PRIMARY).any()
    assert not attempts["policy"].eq(PRIMARY).any()
