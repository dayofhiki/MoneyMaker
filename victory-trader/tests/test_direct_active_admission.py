from __future__ import annotations

import pandas as pd

from victory_trader.direct_active_admission import (
    _diagnose,
    _direct_variant,
)


def test_request153_diagnoses_material_direct_candidate():
    assert _diagnose(0.928, 0.951) == "direct_admission_candidate"


def test_request153_diagnoses_partial_gain_below_target():
    assert _diagnose(0.928, 0.941) == "partial_ranking_improvement"


def test_request153_rejects_immaterial_gain():
    assert _diagnose(0.928, 0.934) == "direct_formulation_insufficient"


def test_direct_variant_replaces_admission_and_balances_retention():
    frame = pd.DataFrame(
        {
            "active_priority": [0.4, 0.2],
            "direct_admission_probability": [0.5, 0.8],
            "transport_survival_probability": [0.6, 0.25],
            "balanced_retention_value": [0.24, 0.05],
        }
    )
    result = _direct_variant(frame)
    assert result["legacy_active_priority"].tolist() == [0.4, 0.2]
    assert result["active_priority"].tolist() == [0.5, 0.8]
    assert result["balanced_retention_value"].tolist() == [0.3, 0.2]
