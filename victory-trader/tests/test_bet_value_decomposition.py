import numpy as np
import pandas as pd

from victory_trader.bet_value_decomposition import (
    _quartiles,
    score_bets,
)


class _Prob:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, x):
        p = np.resize(self.p, len(x))
        return np.column_stack([1.0 - p, p])


class _Reg:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, x):
        return np.full(len(x), self.value, dtype=float)


class _Models:
    direct = _Reg(0.1)
    pwin = _Prob([0.75, 0.25])
    gain = _Reg(4.0)
    loss = _Reg(2.0)
    severe = _Prob([0.2, 0.8])
    gain_cap = 10.0
    loss_cap = 10.0


def test_bet_ev_is_payoff_weighted_not_win_probability_only():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    scored = score_bets(frame, ("x",), _Models())
    assert scored.loc[0, "bet_expected_value_pct"] == 2.5
    assert scored.loc[1, "bet_expected_value_pct"] == -0.5
    assert bool(scored.loc[0, "bet_take"])
    assert not bool(scored.loc[1, "bet_take"])
    assert scored.loc[0, "bet_probability_edge"] > 0
    assert scored.loc[1, "bet_probability_edge"] < 0


def test_quartiles_order_by_predicted_ev():
    frame = pd.DataFrame({
        "bet_expected_value_pct": np.arange(20, dtype=float),
        "trade_return_pct": np.arange(20, dtype=float),
    })
    report = _quartiles(frame)
    assert report["q4"]["realized_trade_mean_pct"] > report["q1"]["realized_trade_mean_pct"]
