import pandas as pd

from victory_trader import fixed_policy_tail_economics as r250


def test_choose_rule_prefers_broadest_then_family_order():
    rows = [
        {"passes": True, "fraction": 0.05, "family": "positive_probability"},
        {"passes": True, "fraction": 0.10, "family": "joint_percentile"},
        {"passes": True, "fraction": 0.10, "family": "predicted_value"},
    ]
    chosen = r250.choose_rule(rows)
    assert chosen["fraction"] == 0.10
    assert chosen["family"] == "predicted_value"


def test_metrics_uses_only_predeclared_tail():
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-05-11"] * 100,
            "fixed_first_watch_value_pct": [1.0] * 10 + [-1.0] * 90,
            "predicted_value": list(range(100)),
            "joint_percentile": [x / 100 for x in range(100)],
            "positive_probability": [x / 100 for x in range(100)],
        }
    )
    row, threshold = r250.metrics(frame, "predicted_value", 0.10)
    assert 89 <= threshold <= 90
    assert row["selected_rows"] >= 10
    assert row["fraction"] == 0.10


def test_request250_contract():
    assert r250.REQUEST_ID == 250
    assert r250.FRACTIONS == (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
    assert r250.MIN_SELECTED_ROWS == 40
    assert r250.MIN_POSITIVE_RATE == 0.35
    assert r250.MIN_POSITIVE_MEAN_DAYS == 5
