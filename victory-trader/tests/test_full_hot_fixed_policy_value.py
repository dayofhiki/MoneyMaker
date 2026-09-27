import numpy as np
import pandas as pd

from victory_trader import full_hot_fixed_policy_value as r249


def test_fixed_target_uses_first_watch_only():
    first_hot=pd.DataFrame([{"trading_day":"2026-06-08","ticker":"TEST","t":0}])
    scan=pd.DataFrame([
        {"trading_day":"2026-06-08","ticker":"TEST","t":60_000,"o":10.0},
        {"trading_day":"2026-06-08","ticker":"TEST","t":120_000,"o":9.0},
        {"trading_day":"2026-06-08","ticker":"TEST","t":180_000,"o":8.0},
        {"trading_day":"2026-06-08","ticker":"TEST","t":240_000,"o":7.0},
        {"trading_day":"2026-06-08","ticker":"TEST","t":300_000,"o":6.0},
        {"trading_day":"2026-06-08","ticker":"TEST","t":360_000,"o":20.0},
        {"trading_day":"2026-06-08","ticker":"TEST","t":420_000,"o":21.0},
    ])
    out=r249.attach_fixed_value(first_hot,scan)
    assert out.loc[0,"fixed_entry_elapsed_minutes"]==1.0
    # The later explosive prices cannot cause the entry to move from the first WATCH.
    assert np.isfinite(out.loc[0,"fixed_first_watch_value_pct"])


def test_request249_contract():
    assert r249.REQUEST_ID==249
    assert r249.EVENT_HORIZONS==(1,2,3,5)
    assert r249.WATCH_WINDOW_MINUTES==5
    assert r249.RISK_CAP_MINUTES==30
