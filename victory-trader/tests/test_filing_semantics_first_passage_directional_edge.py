import math

import pandas as pd

from victory_trader.filing_semantics_first_passage_directional_edge import (
    add_semantic_transforms,
    semantic_coverage,
)


def test_semantic_transforms_match_frozen_definitions():
    frame = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "trading_day": "2026-05-12",
                "filing_semantics_query_success": 1.0,
                "filing_semantics_strict_prior": 1.0,
                "filing_equity_supply_accessions_30d": 2.0,
                "filing_equity_supply_accessions_7d": 1.0,
                "filing_debt_accessions_30d": 1.0,
                "filing_debt_accessions_7d": 0.0,
                "filing_operating_catalyst_accessions_30d": 3.0,
                "filing_operating_catalyst_accessions_7d": 1.0,
                "filing_adverse_accessions_30d": 0.0,
                "filing_adverse_accessions_7d": 0.0,
                "filing_semantic_accessions_30d": 6.0,
                "filing_latest_semantic_age_days": 4.0,
            }
        ]
    )
    result = add_semantic_transforms(frame)
    row = result.iloc[0]
    assert math.isclose(
        row["filing_semantic_log1p_equity_supply_30d"],
        math.log1p(2.0),
    )
    assert math.isclose(
        row["filing_semantic_log1p_latest_age_days"],
        math.log1p(4.0),
    )
    assert math.isclose(
        row["filing_semantic_supply_minus_operating_balance"],
        (2.0 - 3.0) / (2.0 + 3.0 + 1.0),
    )


def test_zero_filing_ticker_is_valid_completed_query():
    frame = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "trading_day": "2026-05-12",
                "filing_semantics_query_success": 1.0,
                "filing_semantics_strict_prior": 1.0,
                "filing_equity_supply_accessions_30d": 0.0,
                "filing_debt_accessions_30d": 0.0,
                "filing_operating_catalyst_accessions_30d": 0.0,
                "filing_adverse_accessions_30d": 0.0,
                "filing_semantic_accessions_30d": 0.0,
            }
        ]
    )
    cov = semantic_coverage(frame)
    assert cov["query_success"] == 1.0
    assert cov["strict_prior"] is True
    assert cov["semantic_30d_rate"] == 0.0
