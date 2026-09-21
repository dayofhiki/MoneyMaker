import numpy as np
import pandas as pd

from victory_trader.expanded_split_supply_hurdle_ev import (
    SPLIT_FEATURES,
    select_policies,
    split_coverage,
    split_supply_feature_frame,
)


def _row(**overrides):
    row = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": 1,
        "entry_price": 5.0,
        "previous_close": 4.0,
        "active_minute_fraction_15m": 1.0,
        "dollar_volume_5m": 200000.0,
        "transactions_5m": 100.0,
        "buy_return_15m_pct": 3.0,
        "buy_return_15m_base_net_return_pct": 1.0,
        "buy_return_15m_stress_net_return_pct": 0.5,
        "supply_hurdle_ev_pct": 0.2,
        "split_hurdle_ev_pct": 0.3,
        "split_history_query_success": 1.0,
        "split_history_strict_prior": 1.0,
        "split_any_730d": 1.0,
        "reverse_split_count_730d": 2.0,
        "reverse_split_count_365d": 1.0,
        "forward_split_count_730d": 0.0,
        "stock_dividend_count_730d": 0.0,
        "split_days_since_latest_reverse": 30.0,
        "split_latest_reverse_consolidation": 10.0,
        "split_max_reverse_consolidation": 20.0,
    }
    row.update(overrides)
    return row


def test_split_feature_frame_appends_exact_preregistered_features():
    frame = pd.DataFrame([_row()])
    features = split_supply_feature_frame(frame)

    for column in SPLIT_FEATURES:
        assert column in features.columns

    assert np.isclose(
        features.iloc[0]["split_log1p_days_since_latest_reverse"],
        np.log1p(30.0),
    )
    assert np.isclose(
        features.iloc[0]["split_log_latest_reverse_consolidation"],
        np.log(10.0),
    )
    assert np.isclose(
        features.iloc[0]["split_log_max_reverse_consolidation"],
        np.log(20.0),
    )


def test_split_feature_frame_keeps_missing_reverse_values_missing():
    frame = pd.DataFrame(
        [
            _row(
                split_any_730d=0.0,
                reverse_split_count_730d=0.0,
                reverse_split_count_365d=0.0,
                split_days_since_latest_reverse=np.nan,
                split_latest_reverse_consolidation=np.nan,
                split_max_reverse_consolidation=np.nan,
            )
        ]
    )
    features = split_supply_feature_frame(frame)

    assert np.isnan(
        features.iloc[0]["split_log1p_days_since_latest_reverse"]
    )
    assert np.isnan(
        features.iloc[0]["split_log_latest_reverse_consolidation"]
    )
    assert np.isnan(
        features.iloc[0]["split_log_max_reverse_consolidation"]
    )


def test_select_policies_uses_zero_semantic_boundaries():
    frame = pd.DataFrame(
        [
            _row(
                ticker="AAA",
                supply_hurdle_ev_pct=0.1,
                split_hurdle_ev_pct=0.2,
            ),
            _row(
                ticker="BBB",
                supply_hurdle_ev_pct=-0.1,
                split_hurdle_ev_pct=0.4,
            ),
            _row(
                ticker="CCC",
                supply_hurdle_ev_pct=0.3,
                split_hurdle_ev_pct=-0.2,
            ),
        ]
    )

    trades, attempts = select_policies(frame)
    counts = attempts.groupby("policy").size().to_dict()

    assert counts["feasible_earliest_15m_cap1"] == 3
    assert counts["supply_hurdle_ev_15m_cap1"] == 2
    assert counts["split_supply_hurdle_ev_15m_cap1"] == 2
    assert len(trades) == 7


def test_split_coverage_reports_support_and_strict_prior():
    frame = pd.DataFrame(
        [
            _row(ticker="AAA", reverse_split_count_730d=1.0),
            _row(
                ticker="BBB",
                split_any_730d=0.0,
                reverse_split_count_730d=0.0,
            ),
        ]
    )
    coverage = split_coverage(frame)

    assert coverage["split_query_success"] == 1.0
    assert coverage["split_strict_prior"] is True
    assert coverage["reverse_split_anchor_rate_730d"] == 0.5
    assert coverage["reverse_split_anchors_730d"] == 1
