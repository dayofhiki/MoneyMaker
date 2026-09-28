import numpy as np
import pandas as pd

from victory_trader import calibrated_execution_admission as r261


def test_request261_contract():
    assert r261.REQUEST_ID == 261
    assert r261.THRESHOLDS == (0.20,0.30,0.40,0.50,0.60,0.70,0.80)
    assert r261.MIN_CAL_PRECISION == 0.55
    assert r261.MIN_CAL_RECALL == 0.25


def test_threshold_selection_uses_calibration_metrics_only():
    frame = pd.DataFrame({
        "execution_supported": [1,1,1,0,0,0,0,0,0,0] * 3,
        "x": np.arange(30),
    })
    class M:
        def predict_proba(self, x):
            p=np.linspace(0.9,0.1,len(x))
            return np.column_stack([1-p,p])
    chosen, rows = r261.choose_threshold(frame, M(), ("x",))
    assert len(rows) == len(r261.THRESHOLDS)
    assert chosen is None or chosen in r261.THRESHOLDS
