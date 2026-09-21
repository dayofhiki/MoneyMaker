import numpy as np
import pandas as pd

from victory_trader.expanded_filing_semantics_hurdle_ev import (
    SEMANTIC_FEATURES,
    filing_semantic_split_supply_feature_frame,
    select_policies,
    semantic_coverage,
)


def _row(**overrides):
    row = {
        "trading_day": "2026-01-02", "ticker": "AAA", "t": 1,
        "entry_price": 5.0, "c": 5.0, "previous_close": 4.0,
        "active_minute_fraction_15m": 1.0, "volume_5m": 50000.0,
        "dollar_volume_5m": 200000.0, "transactions_5m": 100.0,
        "supply_weighted_shares_outstanding_prior": 10000000.0,
        "supply_share_class_shares_outstanding_prior": 8000000.0,
        "buy_return_15m_pct": 3.0,
        "buy_return_15m_base_net_return_pct": 1.0,
        "buy_return_15m_stress_net_return_pct": 0.5,
        "split_hurdle_ev_pct": 0.2, "semantic_hurdle_ev_pct": 0.3,
        "split_history_query_success": 1.0, "split_history_strict_prior": 1.0,
        "split_any_730d": 0.0, "reverse_split_count_730d": 0.0,
        "reverse_split_count_365d": 0.0, "forward_split_count_730d": 0.0,
        "stock_dividend_count_730d": 0.0,
        "split_days_since_latest_reverse": np.nan,
        "split_latest_reverse_consolidation": np.nan,
        "split_max_reverse_consolidation": np.nan,
        "filing_semantics_query_success": 1.0,
        "filing_semantics_strict_prior": 1.0,
        "filing_equity_supply_accessions_30d": 2.0,
        "filing_equity_supply_accessions_7d": 1.0,
        "filing_debt_accessions_30d": 1.0,
        "filing_debt_accessions_7d": 0.0,
        "filing_operating_catalyst_accessions_30d": 1.0,
        "filing_operating_catalyst_accessions_7d": 1.0,
        "filing_adverse_accessions_30d": 0.0,
        "filing_adverse_accessions_7d": 0.0,
        "filing_semantic_accessions_30d": 4.0,
        "filing_latest_semantic_age_days": 2.0,
    }
    row.update(overrides)
    return row


def test_semantic_frame_appends_only_preregistered_features():
    features = filing_semantic_split_supply_feature_frame(pd.DataFrame([_row()]))
    assert all(column in features for column in SEMANTIC_FEATURES)
    assert np.isclose(
        features.iloc[0]["filing_semantic_supply_minus_operating_balance"],
        0.25,
    )


def test_zero_ev_boundaries_are_not_tuned():
    frame = pd.DataFrame([
        _row(ticker="AAA", split_hurdle_ev_pct=0.1, semantic_hurdle_ev_pct=0.2),
        _row(ticker="BBB", split_hurdle_ev_pct=-0.1, semantic_hurdle_ev_pct=0.0),
    ])
    _, attempts = select_policies(frame)
    counts = attempts.groupby("policy").size().to_dict()
    assert counts["feasible_earliest_15m_cap1"] == 2
    assert counts["split_supply_hurdle_ev_15m_cap1"] == 1
    assert counts["filing_semantic_split_supply_hurdle_ev_15m_cap1"] == 1


def test_semantic_coverage_reports_strict_prior():
    result = semantic_coverage(
        pd.DataFrame([_row(), _row(filing_semantic_accessions_30d=0.0)])
    )
    assert result["semantic_query_success"] == 1.0
    assert result["semantic_strict_prior"] is True
    assert result["semantic_30d_rate"] == 0.5
