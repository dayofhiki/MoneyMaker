from __future__ import annotations

import pandas as pd

from victory_trader.economic_full_hot_admission import (
    apply_economic_admission,
)


def test_economic_admission_requires_probability_and_positive_value() -> None:
    frame = pd.DataFrame(
        {
            "tradability_positive_probability": [
                0.80,
                0.80,
                0.60,
                0.90,
            ],
            "predicted_persistent_tradability_pct": [
                0.50,
                -0.10,
                0.70,
                0.00,
            ],
        }
    )
    scored = apply_economic_admission(
        frame,
        probability_gate=0.75,
    )
    assert scored["request232_selected"].tolist() == [
        True,
        True,
        False,
        True,
    ]
    assert scored[
        "economic_admission_selected"
    ].tolist() == [
        True,
        False,
        False,
        False,
    ]
