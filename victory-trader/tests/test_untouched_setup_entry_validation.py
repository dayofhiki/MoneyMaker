import numpy as np

from victory_trader.untouched_setup_entry_validation import (
    _frozen_action_thresholds,
)


def test_request296_uses_only_development_fold_medians():
    payload = {
        "outer_folds": {
            "2026-05-05": {"selected": {
                "utility_threshold": -2.0,
                "turn_threshold": 0.10,
            }},
            "2026-05-06": {"selected": {
                "utility_threshold": -1.0,
                "turn_threshold": 0.20,
            }},
            "2026-05-07": {"selected": {
                "utility_threshold": -3.0,
                "turn_threshold": 0.30,
            }},
            "2026-05-08": {"selected": {
                "utility_threshold": 1.0,
                "turn_threshold": 0.40,
            }},
        }
    }
    result = _frozen_action_thresholds(payload)
    assert np.isclose(result["utility_threshold"], -1.5)
    assert np.isclose(result["turn_threshold"], 0.25)
    assert np.isinf(result["override_threshold"])
