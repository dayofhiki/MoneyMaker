from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.entry_conditioned_rich_second import (
    ENTRY_DELTA_FEATURES,
    attach_entry_conditioned_features,
)
from victory_trader.rich_second_position_value import (
    RICH_SECOND_FEATURES,
)


def _frame() -> pd.DataFrame:
    rows = []
    for i, value in enumerate([0.2, 0.5, -0.1]):
        row = {
            "trading_day": "2026-05-01",
            "ticker": "AAA",
            "hot_t": 1,
            "state_t": (i + 1) * 60_000,
        }
        for source in RICH_SECOND_FEATURES:
            row[source] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_entry_delta_is_anchored_to_first_observation() -> None:
    enriched = attach_entry_conditioned_features(_frame())
    source = RICH_SECOND_FEATURES[0]
    column = f"entry_delta_{source}"
    assert np.isclose(enriched.iloc[0][column], 0.0)
    assert np.isclose(enriched.iloc[1][column], 0.3)
    assert np.isclose(enriched.iloc[2][column], -0.3)


def test_future_change_cannot_change_earlier_entry_delta() -> None:
    base = attach_entry_conditioned_features(_frame())
    changed = _frame()
    source = RICH_SECOND_FEATURES[0]
    changed.loc[2, source] = 99.0
    changed = attach_entry_conditioned_features(changed)
    column = f"entry_delta_{source}"
    assert np.isclose(base.iloc[1][column], changed.iloc[1][column])


def test_missing_entry_value_does_not_use_future_anchor() -> None:
    frame = _frame()
    source = RICH_SECOND_FEATURES[0]
    frame.loc[0, source] = np.nan
    enriched = attach_entry_conditioned_features(frame)
    assert enriched[f"entry_delta_{source}"].isna().all()


def test_entry_delta_family_covers_all_rich_sources() -> None:
    expected = {
        f"entry_delta_{source}"
        for source in RICH_SECOND_FEATURES
    }
    assert set(ENTRY_DELTA_FEATURES) == expected
