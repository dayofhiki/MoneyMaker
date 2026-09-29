import numpy as np
import pandas as pd

from victory_trader import fixed_horizon_position_signal as r248


def test_fixed_horizon_never_selects_best_future():
    states=pd.DataFrame([
        {"trading_day":"2026-05-05","ticker":"TEST","hot_t":0,"state_t":60_000,"terminal":False,"attention_score":1.0},
        {"trading_day":"2026-05-05","ticker":"TEST","hot_t":0,"state_t":120_000,"terminal":False,"attention_score":1.0},
        {"trading_day":"2026-05-05","ticker":"TEST","hot_t":0,"state_t":180_000,"terminal":False,"attention_score":1.0},
        {"trading_day":"2026-05-05","ticker":"TEST","hot_t":0,"state_t":240_000,"terminal":False,"attention_score":1.0},
    ])
    times=np.array([61_000,121_000,181_000,241_000],dtype=np.int64)
    opens=np.array([10.0,9.0,20.0,8.0])
    paths={("2026-05-05","TEST"):(times,opens)}
    one=r248.horizon_targets(states,np.ones(4),paths,1)
    two=r248.horizon_targets(states,np.ones(4),paths,2)
    assert one.iloc[0].target < 0
    assert two.iloc[0].target > 0


def test_request248_contract_is_frozen():
    assert r248.REQUEST_ID==248
    assert r248.HORIZONS==(1,2,3,5)
    assert r248.ADMISSION_THRESHOLD==0.60
    assert r248.MIN_SIGN_AUC==0.55
    assert r248.MIN_SPEARMAN==0.05
    assert r248.MIN_SPREAD==0.30
