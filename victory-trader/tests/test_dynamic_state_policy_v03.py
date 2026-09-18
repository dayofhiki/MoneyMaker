import pandas as pd

from victory_trader.dynamic_state_policy_v03 import (
    _exit_signal,
    _position_fraction,
    simulate_one_position,
)


def test_position_fraction_matches_half_percent_risk_on_five_percent_stop():
    assert abs(_position_fraction() - 0.10) < 1e-12


def test_min_hold_prevents_soft_exit_before_five_minutes():
    row = pd.Series(
        {
            "c": 10.0,
            "hod_distance_pct": -1.0,
            "above_regular_vwap": True,
            "predicted_mean_gross_10m_pct": -0.5,
            "predicted_severe_mae_prob": 0.9,
        }
    )
    assert _exit_signal(row, entry_reference=10.0, hold_minutes=2.0) is None
    assert (
        _exit_signal(row, entry_reference=10.0, hold_minutes=5.0)
        == "expected_gross_nonpositive"
    )


def test_hard_stop_can_exit_before_min_hold():
    row = pd.Series(
        {
            "c": 9.4,
            "hod_distance_pct": -2.0,
            "above_regular_vwap": True,
            "predicted_mean_gross_10m_pct": 1.0,
            "predicted_severe_mae_prob": 0.1,
        }
    )
    assert _exit_signal(row, entry_reference=10.0, hold_minutes=1.0) == "hard_stop"


def test_one_position_policy_enters_and_exits_on_expected_value_decay():
    rows = []
    for minute in range(8):
        t = minute * 60_000
        pred_base = 0.8 if minute <= 5 else -0.2
        pred_gross = 1.0 if minute <= 5 else -0.1
        rows.append(
            {
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "t": t,
                "c": 10.0 + minute * 0.02,
                "entry_price": 10.01 + minute * 0.02,
                "hod_distance_pct": -1.0,
                "above_regular_vwap": True,
                "predicted_mean_base_10m_pct": pred_base,
                "predicted_mean_gross_10m_pct": pred_gross,
                "predicted_severe_mae_prob": 0.1,
            }
        )
    frame = pd.DataFrame(rows)
    trades, equity = simulate_one_position(frame, entry_ev_gate=0.25)
    assert len(trades) == 1
    assert trades.iloc[0]["hold_minutes"] >= 5
    assert trades.iloc[0]["exit_reason"] == "expected_gross_nonpositive"
    assert not equity.empty
    assert trades.iloc[0]["position_fraction"] == 0.10


def test_failure_state_obeys_minimum_hold_hysteresis():
    row = pd.Series(
        {
            "c": 9.8,
            "hod_distance_pct": -6.0,
            "above_regular_vwap": False,
            "predicted_mean_gross_10m_pct": 0.5,
            "predicted_severe_mae_prob": 0.2,
        }
    )
    assert _exit_signal(row, entry_reference=10.0, hold_minutes=2.0) is None
    assert (
        _exit_signal(row, entry_reference=10.0, hold_minutes=5.0)
        == "failure_state"
    )
