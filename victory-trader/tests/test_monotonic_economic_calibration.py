from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.monotonic_economic_calibration import (
    _add_selection,
    _fit_isotonic,
)


def test_isotonic_map_is_monotonic() -> None:
    frame = pd.DataFrame(
        {
            "trading_day": [
                "2026-05-11",
                "2026-05-11",
                "2026-05-12",
                "2026-05-12",
            ]
            * 250,
            "predicted_persistent_tradability_pct": [
                -2.0,
                -1.0,
                1.0,
                2.0,
            ]
            * 250,
            "persistent_tradability_pct": [
                -3.0,
                -1.0,
                0.5,
                1.5,
            ]
            * 250,
        }
    )
    model = _fit_isotonic(frame)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = model.predict(x)
    assert np.all(np.diff(y) >= 0)


def test_selection_requires_p75_and_positive_calibrated_value() -> None:
    frame = pd.DataFrame(
        {
            "tradability_selected": [
                True,
                True,
                False,
                True,
            ],
            "mapped": [0.2, -0.1, 1.0, 0.0],
        }
    )
    result = _add_selection(
        frame,
        "mapped",
        "selected",
    )
    assert result["selected"].tolist() == [
        True,
        False,
        False,
        False,
    ]
