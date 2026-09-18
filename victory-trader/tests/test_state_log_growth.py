import numpy as np
import pandas as pd
import pytest

from victory_trader.state_log_growth import (
    LOG_GROWTH_GATE,
    _select_first_trades,
    arithmetic_to_log_growth,
)


def _row(
    minute: int,
    *,
    predicted_log_growth: float,
    horizon: int = 15,
    entry_price: float = 5.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "predicted_best_log_growth": predicted_log_growth,
        "predicted_best_log_growth_horizon_min": horizon,
        "predicted_best_base_ev_pct": 0.0,
        "predicted_best_horizon_min": horizon,
    }
    for current in (5, 10, 15):
        row[f"buy_return_{current}m_base_net_return_pct"] = current / 10
        row[f"buy_return_{current}m_pct"] = current / 10 + 1.0
        row[f"buy_return_{current}m_stress_net_return_pct"] = current / 10 - 1.0
    return row


def test_log_growth_gate_matches_half_percent_wealth_hurdle():
    expected = arithmetic_to_log_growth(np.asarray([0.5]))[0]
    assert LOG_GROWTH_GATE == pytest.approx(expected)


def test_log_growth_penalizes_equal_sized_loss_more_than_gain_repairs():
    positive, negative = arithmetic_to_log_growth(
        np.asarray([50.0, -50.0])
    )
    assert positive > 0
    assert negative < 0
    assert positive + negative < 0


def test_log_growth_rejects_total_or_worse_loss():
    with pytest.raises(ValueError, match="<= -100%"):
        arithmetic_to_log_growth(np.asarray([-100.0]))

    with pytest.raises(ValueError, match="<= -100%"):
        arithmetic_to_log_growth(np.asarray([-105.0]))


def test_log_growth_policy_enters_first_gate_crossing():
    frame = pd.DataFrame(
        [
            _row(0, predicted_log_growth=LOG_GROWTH_GATE - 1e-6),
            _row(
                1,
                predicted_log_growth=LOG_GROWTH_GATE,
                horizon=10,
            ),
            _row(
                2,
                predicted_log_growth=LOG_GROWTH_GATE + 1.0,
                horizon=15,
            ),
        ]
    )
    trades = _select_first_trades(frame, policy="log_growth_cap1")

    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000
    assert int(trades.iloc[0]["action_horizon_min"]) == 10
    assert trades.iloc[0]["realized_base_net_return_pct"] == 1.0


def test_first_log_signal_is_consumed_when_unfillable():
    frame = pd.DataFrame(
        [
            _row(
                0,
                predicted_log_growth=LOG_GROWTH_GATE + 0.1,
                entry_price=np.nan,
            ),
            _row(
                1,
                predicted_log_growth=LOG_GROWTH_GATE + 1.0,
            ),
        ]
    )
    trades = _select_first_trades(frame, policy="log_growth_cap1")
    assert trades.empty
