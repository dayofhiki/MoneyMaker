import pandas as pd
import pytest

from victory_trader.shortlist_risk_audit import risk_diagnostics


def sample():
    return pd.DataFrame([
        {
            "resolved": True,
            "trade_return_pct": -4.0,
            "severe_probability": 0.90,
            "severe_threshold": 0.50,
        },
        {
            "resolved": True,
            "trade_return_pct": -3.0,
            "severe_probability": 0.80,
            "severe_threshold": 0.50,
        },
        {
            "resolved": True,
            "trade_return_pct": 1.0,
            "severe_probability": 0.20,
            "severe_threshold": 0.50,
        },
        {
            "resolved": True,
            "trade_return_pct": 2.0,
            "severe_probability": 0.10,
            "severe_threshold": 0.50,
        },
    ])


def test_risk_diagnostic_separates_rejected_and_accepted():
    result = risk_diagnostics(sample())
    assert result["auc"] == pytest.approx(1.0)
    assert result["rejected_rows"] == 2
    assert result["accepted_rows"] == 2
    assert result["rejected_severe_rate"] == pytest.approx(1.0)
    assert result["accepted_severe_rate"] == pytest.approx(0.0)
    assert result["accepted_mean_advantage_pct"] == pytest.approx(5.0)


def test_unresolved_rows_are_not_scored_as_safe():
    frame = pd.concat([
        sample(),
        pd.DataFrame([{
            "resolved": False,
            "trade_return_pct": float("nan"),
            "severe_probability": 0.99,
            "severe_threshold": 0.50,
        }]),
    ], ignore_index=True)
    result = risk_diagnostics(frame)
    assert result["resolved_rows"] == 4


def test_average_precision_lift_uses_severe_prevalence_baseline():
    result = risk_diagnostics(sample())
    assert result["severe_prevalence"] == pytest.approx(0.5)
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["average_precision_lift_vs_prevalence"] == pytest.approx(0.5)
