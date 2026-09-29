from __future__ import annotations
import numpy as np
import pandas as pd
from victory_trader.continuation_availability import attach_labels

def test_availability_is_defined_even_when_values_missing() -> None:
    frame = pd.DataFrame({
        "watch_event_1_multi_event_value_pct":[1.0, np.nan],
        "watch_event_2_multi_event_value_pct":[2.0, np.nan],
        "watch_event_3_multi_event_value_pct":[np.nan, np.nan],
        "watch_event_4_multi_event_value_pct":[np.nan, np.nan],
        "watch_event_5_multi_event_value_pct":[np.nan, np.nan],
    })
    out=attach_labels(frame)
    assert out["continuation_available"].tolist()==[1,0]
    assert out["available_watch_count"].tolist()==[2,0]
    assert np.isclose(out.loc[0,"conditional_top2_value_pct"],1.5)
    assert pd.isna(out.loc[1,"conditional_top2_value_pct"])
