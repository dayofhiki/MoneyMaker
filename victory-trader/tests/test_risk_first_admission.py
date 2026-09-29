import pandas as pd
import pytest

from victory_trader.risk_first_admission import (
    admission_mask,
    decision_metrics,
)


def predictions():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 1,
            "predicted_return_pct": 1.0,
            "severe_probability": 0.1,
        }
    ])


def policy(resolved=False):
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 1,
            "entered": True,
            "resolved": resolved,
            "trade_return_pct": (
                1.0 if resolved else float("nan")
            ),
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "BBB",
            "hot_t": 2,
            "entered": False,
            "resolved": True,
            "trade_return_pct": float("nan"),
        },
    ])


def test_admission_requires_low_risk_and_high_value():
    frame = pd.DataFrame([
        {"severe_probability": 0.1, "predicted_return_pct": 1.0},
        {"severe_probability": 0.9, "predicted_return_pct": 1.0},
        {"severe_probability": 0.1, "predicted_return_pct": -1.0},
    ])
    mask = admission_mask(
        frame,
        risk_threshold=0.5,
        value_threshold=0.0,
    )
    assert mask.tolist() == [True, False, False]


def test_rejected_unresolved_pullback_becomes_cash():
    metrics = decision_metrics(
        policy(resolved=False),
        predictions(),
        risk_threshold=0.05,
        value_threshold=0.0,
    )
    assert metrics["admissions"] == 0
    assert metrics["resolution_or_cash_rate"] == pytest.approx(1.0)
    assert metrics["candidate_mean_pct"] == pytest.approx(0.0)


def test_admitted_unresolved_pullback_stays_unresolved():
    metrics = decision_metrics(
        policy(resolved=False),
        predictions(),
        risk_threshold=0.5,
        value_threshold=0.0,
    )
    assert metrics["admissions"] == 1
    assert metrics["resolution_or_cash_rate"] == pytest.approx(0.5)


def test_admitted_resolved_trade_keeps_economic_return():
    metrics = decision_metrics(
        policy(resolved=True),
        predictions(),
        risk_threshold=0.5,
        value_threshold=0.0,
    )
    assert metrics["trade_mean_pct"] == pytest.approx(1.0)
    assert metrics["candidate_mean_pct"] == pytest.approx(0.5)
