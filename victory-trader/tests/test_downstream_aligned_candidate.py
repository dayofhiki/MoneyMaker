import numpy as np
import pandas as pd
import pytest

from victory_trader.downstream_aligned_candidate import (
    attach_downstream_target,
    candidate_columns,
    selection_metrics,
    threshold_from_training,
)


def first_hot():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "aaa",
            "t": 100,
            "feature_x": 1.0,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "bbb",
            "t": 200,
            "feature_x": 2.0,
        },
    ])


def policy():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 100,
            "entered": False,
            "resolved": True,
            "economic_return_pct": 0.0,
            "trade_return_pct": np.nan,
            "exit_t": None,
            "exit_reason": "cash_no_pullback",
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "BBB",
            "hot_t": 200,
            "entered": True,
            "resolved": False,
            "economic_return_pct": np.nan,
            "trade_return_pct": np.nan,
            "exit_t": None,
            "exit_reason": "missing_deadline_state",
        },
    ])


def test_no_pullback_is_cash_zero_and_unresolved_trade_stays_unknown():
    labeled = attach_downstream_target(first_hot(), policy())
    aaa = labeled.loc[labeled.ticker.eq("AAA")].iloc[0]
    bbb = labeled.loc[labeled.ticker.eq("BBB")].iloc[0]
    assert aaa.downstream_value_pct == pytest.approx(0.0)
    assert pd.isna(bbb.downstream_value_pct)


def test_candidate_feature_contract_rejects_target_columns(monkeypatch):
    import victory_trader.downstream_aligned_candidate as module

    monkeypatch.setattr(
        module,
        "usable_columns",
        lambda frame: (
            "some_feature",
            "downstream_value_pct",
        ),
    )
    with pytest.raises(ValueError, match="target leakage"):
        candidate_columns(pd.DataFrame({"some_feature": [1.0]}))


def test_threshold_is_training_quantile_only():
    values = np.arange(100, dtype=float)
    threshold = threshold_from_training(values, fraction=0.20)
    assert threshold == pytest.approx(np.quantile(values, 0.80))


def test_unselected_unresolved_candidate_becomes_cash():
    predictions = pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 100,
            "selected_aligned": False,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "BBB",
            "hot_t": 200,
            "selected_aligned": False,
        },
    ])
    metrics = selection_metrics(
        policy(),
        predictions,
        "selected_aligned",
    )
    assert metrics["selected_candidates"] == 0
    assert metrics["resolution_or_cash_rate"] == pytest.approx(1.0)
    assert metrics["candidate_mean_pct"] == pytest.approx(0.0)


def test_selected_unresolved_entry_remains_unresolved():
    predictions = pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 100,
            "selected_aligned": False,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "BBB",
            "hot_t": 200,
            "selected_aligned": True,
        },
    ])
    metrics = selection_metrics(
        policy(),
        predictions,
        "selected_aligned",
    )
    assert metrics["selected_candidates"] == 1
    assert metrics["unresolved_selected_entries"] == 1
    assert metrics["resolution_or_cash_rate"] == pytest.approx(0.5)
