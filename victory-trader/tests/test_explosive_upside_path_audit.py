import numpy as np
import pandas as pd

from victory_trader.explosive_upside_path_audit import (
    excursion_for_horizon,
)


def test_excursion_audit_tracks_upside_and_prepeak_adverse_move():
    entry_t = 1_000_000
    minute = 60_000
    path = pd.DataFrame({
        "t": [
            entry_t + minute,
            entry_t + 2 * minute,
            entry_t + 3 * minute,
            entry_t + 10 * minute,
        ],
        "o": [99.0, 102.0, 108.0, 107.0],
        "h": [101.0, 104.0, 111.0, 109.0],
        "l": [97.0, 100.0, 105.0, 106.0],
    })
    out = excursion_for_horizon(
        path,
        entry_t=entry_t,
        entry_price=100.0,
        horizon_minutes=10,
    )
    assert out["complete"]
    assert np.isclose(out["mfe_high_pct"], 11.0)
    assert np.isclose(out["mfe_open_pct"], 8.0)
    assert np.isclose(out["mae_low_pct"], -3.0)
    assert np.isclose(out["pre_high_peak_mae_low_pct"], -3.0)
    assert out["first_high_hit_10_minutes"] == 3.0
