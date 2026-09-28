from victory_trader import causal_opportunity_ceiling as r265

def test_request265_contract():
    assert r265.REQUEST_ID == 265
    assert r265.MIN_ORACLE_MEAN == 0.50
    assert r265.MIN_ANY_POSITIVE_RATE == 0.50
    assert r265.MIN_SPEARMAN == 0.08
    assert r265.MIN_AUC == 0.58
