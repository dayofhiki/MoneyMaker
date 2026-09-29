import numpy as np
import pandas as pd

from victory_trader.asymmetric_tail_bet_value import (
    regime_labels,
    score_tail_bets,
)


class _Direct:
    def predict(self, x):
        return np.zeros(len(x), dtype=float)


class _Regime:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, x):
        rows = [
            [0.10, 0.20, 0.70],
            [0.70, 0.20, 0.10],
        ]
        return np.asarray(rows[:len(x)], dtype=float)


class _Models:
    direct = _Direct()
    regime = _Regime()
    payoffs = (-4.0, 0.0, 6.0)


def test_regime_labels_use_symmetric_tail_thresholds():
    values = np.array([-3.0, -1.0, 0.0, 1.5, 2.0, 5.0])
    assert regime_labels(values).tolist() == [0, 1, 1, 1, 2, 2]


def test_tail_ev_prices_upside_and_downside():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    scored = score_tail_bets(frame, ("x",), _Models())
    assert np.isclose(scored.loc[0, "tail_expected_value_pct"], 3.8)
    assert np.isclose(scored.loc[1, "tail_expected_value_pct"], -2.2)
    assert bool(scored.loc[0, "tail_take"])
    assert not bool(scored.loc[1, "tail_take"])
