import pandas as pd
import pytest

from victory_trader.joint_pullback_admission import (
    build_pullback_state_rows,
    decision_metrics,
)


def scored():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "t": 0,
            "candidate_probability": 0.7,
        }
    ])


def scan(prices):
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "t": (i + 1) * 60_000,
            "o": price,
        }
        for i, price in enumerate(prices)
    ])


def test_first_two_percent_pullback_is_the_only_state():
    rows = build_pullback_state_rows(
        scored(),
        scan([10.0, 10.4, 10.3, 10.19, 9.9]),
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row.entry_t == 240_000
    assert row.entry_price == pytest.approx(10.19)
    assert row.drawdown_from_running_high_pct <= -2.0


def test_future_prices_cannot_change_admission_features():
    prefix = [10.0, 10.4, 10.3, 10.19]
    left = build_pullback_state_rows(
        scored(),
        scan(prefix),
    )
    right = build_pullback_state_rows(
        scored(),
        scan(prefix + [8.0, 12.0, 7.0]),
    )
    columns = [
        "entry_t",
        "entry_price",
        "elapsed_minutes",
        "event_index",
        "current_return_from_first_pct",
        "running_high_return_from_first_pct",
        "running_low_return_from_first_pct",
        "drawdown_from_running_high_pct",
        "recovery_from_running_low_pct",
        "recovery_fraction_of_range",
        "last_event_return_pct",
        "two_event_return_pct",
        "path_range_pct",
        "path_efficiency",
        "events_since_high",
        "events_since_low",
    ]
    pd.testing.assert_series_equal(
        left.iloc[0][columns],
        right.iloc[0][columns],
        check_names=False,
    )


def test_no_pullback_produces_no_admission_state():
    rows = build_pullback_state_rows(
        scored(),
        scan([10.0, 10.1, 10.2, 10.3]),
    )
    assert rows.empty


def policy_rows(resolved=False):
    return pd.DataFrame([
        {
            "trading_day": "2026-05-08",
            "ticker": "AAA",
            "hot_t": 1,
            "entered": True,
            "resolved": resolved,
            "trade_return_pct": (
                1.0 if resolved else float("nan")
            ),
        },
        {
            "trading_day": "2026-05-08",
            "ticker": "BBB",
            "hot_t": 2,
            "entered": False,
            "resolved": True,
            "trade_return_pct": float("nan"),
        },
    ])


def predicted(prob=0.9, value=1.0):
    return pd.DataFrame([
        {
            "trading_day": "2026-05-08",
            "ticker": "AAA",
            "hot_t": 1,
            "positive_probability": prob,
            "predicted_return_pct": value,
        }
    ])


def test_rejected_unresolved_trade_becomes_cash():
    metrics = decision_metrics(
        policy_rows(resolved=False),
        predicted(prob=0.1, value=-1.0),
        probability_threshold=0.5,
        require_positive_value=True,
    )
    assert metrics["admissions"] == 0
    assert metrics["resolution_or_cash_rate"] == pytest.approx(1.0)
    assert metrics["candidate_mean_pct"] == pytest.approx(0.0)


def test_admitted_unresolved_trade_stays_unresolved():
    metrics = decision_metrics(
        policy_rows(resolved=False),
        predicted(),
        probability_threshold=0.5,
        require_positive_value=True,
    )
    assert metrics["admissions"] == 1
    assert metrics["resolution_or_cash_rate"] == pytest.approx(0.5)
