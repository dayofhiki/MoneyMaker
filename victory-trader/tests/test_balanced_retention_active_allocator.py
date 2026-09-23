from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.balanced_retention_active_allocator import (
    add_balanced_retention_value,
)


def test_balanced_retention_value_requires_both_opportunity_and_survival():
    frame = pd.DataFrame(
        {
            "active_priority": [0.8, 0.8, 0.2],
            "transport_survival_probability": [0.9, 0.1, 0.9],
        }
    )

    result = add_balanced_retention_value(frame)

    np.testing.assert_allclose(
        result["balanced_retention_value"].to_numpy(),
        np.array([0.72, 0.08, 0.18]),
    )
    assert (
        result.loc[0, "balanced_retention_value"]
        > result.loc[2, "balanced_retention_value"]
        > result.loc[1, "balanced_retention_value"]
    )
