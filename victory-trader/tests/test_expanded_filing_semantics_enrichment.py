from datetime import date

import pandas as pd

from victory_trader.expanded_filing_semantics_enrichment import (
    enrich_frame,
    semantic_features,
)


def _row(day, accession, secondary, tertiary):
    return {
        "_filing_day": date.fromisoformat(day),
        "accession_number": accession,
        "secondary_category": secondary,
        "tertiary_category": tertiary,
    }


def test_semantic_features_are_strictly_prior_and_deduplicate_accessions():
    rows = [
        _row("2026-01-01", "A1", "equity_activity", "public_offering"),
        _row("2026-01-01", "A1", "equity_activity", "underwriting_agreement"),
        _row("2025-12-20", "A2", "debt_activity", "credit_facility"),
        _row("2026-01-02", "A3", "business_developments", "clinical_trial_results"),
    ]
    result = semantic_features(rows, date(2026, 1, 2))
    assert result["filing_equity_supply_accessions_7d"] == 1.0
    assert result["filing_debt_accessions_30d"] == 1.0
    assert result["filing_operating_catalyst_accessions_30d"] == 0.0
    assert result["filing_latest_semantic_age_days"] == 1.0


def test_enrichment_preserves_valid_zero_filing_state():
    frame = pd.DataFrame(
        [{"ticker": "AAA", "trading_day": "2026-01-02", "t": 1}]
    )
    enriched = enrich_frame(frame, {"AAA": []}, {"AAA"})
    assert enriched.iloc[0]["filing_semantics_query_success"] == 1.0
    assert enriched.iloc[0]["filing_semantic_accessions_30d"] == 0.0

