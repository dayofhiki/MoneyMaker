import pandas as pd
import pytest

from victory_trader.two_stage_risk_veto import two_stage_metrics


def policy():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 1,
            "entered": True,
            "resolved": True,
            "economic_return_pct": -3.0,
            "trade_return_pct": -3.0,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "BBB",
            "hot_t": 2,
            "entered": True,
            "resolved": True,
            "economic_return_pct": 2.0,
            "trade_return_pct": 2.0,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "CCC",
            "hot_t": 3,
            "entered": False,
            "resolved": True,
            "economic_return_pct": 0.0,
            "trade_return_pct": None,
        },
    ])


def prehot():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 1,
            "selected_prehot": True,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "BBB",
            "hot_t": 2,
            "selected_prehot": True,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "CCC",
            "hot_t": 3,
            "selected_prehot": True,
        },
    ])


def risk():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 1,
            "risk_accepted": False,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "BBB",
            "hot_t": 2,
            "risk_accepted": True,
        },
    ])


def test_risk_veto_turns_rejected_trade_into_cash():
    metrics = two_stage_metrics(
        policy(),
        prehot(),
        risk(),
    )
    assert metrics["shortlisted_pullback_triggers"] == 2
    assert metrics["risk_rejected_triggers"] == 1
    assert metrics["trade_count"] == 1
    assert metrics["candidate_mean_pct"] == pytest.approx(2.0 / 3.0)
    assert metrics["trade_mean_pct"] == pytest.approx(2.0)
    assert metrics["trade_severe_loss_rate_le_minus2"] == pytest.approx(0.0)
