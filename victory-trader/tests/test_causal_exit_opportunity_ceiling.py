from victory_trader import causal_exit_opportunity_ceiling as r266

def test_request266_contract():
    assert r266.REQUEST_ID == 266
    assert r266.ENTRY_WINDOW_MINUTES == 10
    assert r266.EXIT_CAP_MINUTES == 30
