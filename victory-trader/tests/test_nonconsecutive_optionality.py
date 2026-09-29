from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.nonconsecutive_optionality import attach_target


def test_target_uses_best_two_even_when_not_adjacent() -> None:
    frame = pd.DataFrame(
        {
            "watch_event_1_multi_event_value_pct": [2.0],
            "watch_event_2_multi_event_value_pct": [-3.0],
            "watch_event_3_multi_event_value_pct": [1.0],
            "watch_event_4_multi_event_value_pct": [-2.0],
            "watch_event_5_multi_event_value_pct": [0.0],
        }
    )
    out = attach_target(frame)
    assert bool(out.loc[0, "nonconsecutive_evaluable"])
    assert np.isclose(
        out.loc[0, "nonconsecutive_top2_tradability_pct"],
        1.5,
    )


def test_target_needs_two_watch_opportunities() -> None:
    frame = pd.DataFrame(
        {
            "watch_event_1_multi_event_value_pct": [2.0],
            "watch_event_2_multi_event_value_pct": [np.nan],
            "watch_event_3_multi_event_value_pct": [np.nan],
            "watch_event_4_multi_event_value_pct": [np.nan],
            "watch_event_5_multi_event_value_pct": [np.nan],
        }
    )
    out = attach_target(frame)
    assert not bool(out.loc[0, "nonconsecutive_evaluable"])
    assert pd.isna(
        out.loc[0, "nonconsecutive_top2_tradability_pct"]
    )
