import numpy as np
import pandas as pd

from victory_trader.fresh_crack_entry_validation import (
    FRESH_DAYS,
    FROZEN_OVERRIDE_THRESHOLD,
    FROZEN_TURN_THRESHOLD,
    FROZEN_UTILITY_THRESHOLD,
    _setup_feature_contract,
)


def test_frozen_thresholds_are_single_values_and_override_disabled():
    assert np.isclose(FROZEN_UTILITY_THRESHOLD, -2.00886730522822)
    assert np.isclose(FROZEN_TURN_THRESHOLD, 0.12298633907918213)
    assert np.isinf(FROZEN_OVERRIDE_THRESHOLD)
    assert len(FRESH_DAYS) == 5


def test_setup_contract_uses_only_prehot_context_extension():
    frame = pd.DataFrame({
        "attention_score": [1.0],
        "return_from_previous_close_pct": [2.0],
        "prehot_lag1_attention_score": [0.5],
        "prehot_delta1_attention_score": [0.5],
        "crack_mfe_open_60m_pct": [10.0],
        "future_cheat": [99.0],
    })
    base, rich = _setup_feature_contract(frame)
    assert "attention_score" in base
    assert "prehot_lag1_attention_score" in rich
    assert "prehot_delta1_attention_score" in rich
    assert "crack_mfe_open_60m_pct" not in rich
    assert "future_cheat" not in rich
