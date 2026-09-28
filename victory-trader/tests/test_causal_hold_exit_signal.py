from victory_trader import causal_hold_exit_signal as r267

def test_request267_contract():
    assert r267.REQUEST_ID == 267
    assert r267.POSITION_CAP_MINUTES == 30
    assert r267.MIN_SPEARMAN == 0.10
    assert r267.MIN_AUC == 0.60
