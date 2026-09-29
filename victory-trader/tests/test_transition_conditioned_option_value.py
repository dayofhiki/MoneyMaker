from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.transition_conditioned_option_value import (
    PHASES,
    attach_transition_phase,
)


def _frame() -> pd.DataFrame:
    # current change and acceleration imply:
    # previous = current - acceleration
    return pd.DataFrame(
        {
            "position_return_change_1m_pct": [
                1.0,   # prev -0.5 -> reacceleration
                -1.0,  # prev +0.5 -> pullback onset
                1.0,   # prev +0.5 -> continuing strength
                -1.0,  # prev -0.5 -> continuing weakness
                np.nan,
            ],
            "position_return_accel_1m_pct": [
                1.5,
                -1.5,
                0.5,
                -0.5,
                np.nan,
            ],
        }
    )


def test_transition_phase_sign_geometry() -> None:
    enriched = attach_transition_phase(_frame())
    assert enriched["transition_phase"].tolist() == [
        "reacceleration",
        "pullback_onset",
        "continuing_strength",
        "continuing_weakness",
        "flat_or_sparse",
    ]


def test_previous_move_is_reconstructed_causally() -> None:
    enriched = attach_transition_phase(_frame())
    expected = [-0.5, 0.5, 0.5, -0.5]
    actual = enriched.loc[
        :3, "previous_observed_return_change_pct"
    ].to_numpy(dtype=float)
    assert np.allclose(actual, expected)


def test_future_rows_do_not_change_earlier_phase() -> None:
    base = attach_transition_phase(_frame())
    changed = _frame()
    changed.loc[4, "position_return_change_1m_pct"] = 99.0
    changed.loc[4, "position_return_accel_1m_pct"] = 100.0
    changed = attach_transition_phase(changed)
    assert (
        base.loc[:3, "transition_phase"].tolist()
        == changed.loc[:3, "transition_phase"].tolist()
    )


def test_phase_catalog_is_frozen() -> None:
    assert PHASES == (
        "reacceleration",
        "pullback_onset",
        "continuing_strength",
        "continuing_weakness",
        "flat_or_sparse",
    )
