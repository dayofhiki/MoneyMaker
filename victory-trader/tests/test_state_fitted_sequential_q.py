import numpy as np
import pandas as pd

from victory_trader.state_fitted_sequential_q import (
    QModel,
    SequentialModels,
    _choose_mid_action,
    checkpoint_frames,
    select_sequential_policy,
    split_training_days,
)


MINUTE_MS = 60_000


class ConstantPredictor:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def _q(value):
    return QModel(
        model=ConstantPredictor(value),
        feature_columns=tuple(),
        target_low=-100.0,
        target_high=100.0,
    )


def _row(minute, *, base10=1.0, day="2026-01-02", ticker="AAA"):
    return {
        "trading_day": day,
        "ticker": ticker,
        "t": minute * MINUTE_MS,
        "c": 5.0,
        "previous_close": 4.0,
        "active_minute_fraction_15m": 1.0,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": 5.0,
        "buy_return_10m_base_net_return_pct": base10,
        "buy_return_10m_pct": base10 + 1.0,
        "buy_return_10m_stress_net_return_pct": base10 - 1.0,
    }


def test_split_training_days_is_chronological_and_disjoint():
    frame = pd.DataFrame(
        [
            _row(0, day=f"2026-01-{day:02d}")
            for day in range(1, 10)
        ]
    )
    a, b, c = split_training_days(frame)

    assert not (a & b)
    assert not (a & c)
    assert not (b & c)
    assert a | b | c == set(frame["trading_day"])
    assert max(a) < min(b)
    assert max(b) < min(c)


def test_checkpoint_frames_uses_exact_zero_five_ten_offsets():
    frame = pd.DataFrame(
        [
            _row(1, base10=1.0),
            _row(5, base10=99.0),
            _row(6, base10=2.0),
            _row(11, base10=3.0),
        ]
    )
    checkpoints = checkpoint_frames(frame)

    assert int(checkpoints[0].iloc[0]["t"]) == 1 * MINUTE_MS
    assert int(checkpoints[5].iloc[0]["t"]) == 6 * MINUTE_MS
    assert int(checkpoints[10].iloc[0]["t"]) == 11 * MINUTE_MS


def test_mid_action_exact_tie_order_skip_wait_buy():
    assert _choose_mid_action(0.0, 0.0) == "SKIP"
    assert _choose_mid_action(1.0, 1.0) == "WAIT"
    assert _choose_mid_action(2.0, 1.0) == "BUY"


def test_sequential_policy_can_wait_then_buy():
    frame = pd.DataFrame(
        [
            _row(0, base10=-1.0),
            _row(5, base10=3.0),
            _row(10, base10=2.0),
        ]
    )
    models = SequentialModels(
        q10_buy=_q(1.0),
        q5_buy=_q(2.0),
        q5_wait=_q(1.0),
        q0_buy=_q(-1.0),
        q0_wait=_q(1.0),
    )

    trades, paths = select_sequential_policy(frame, models)

    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 5 * MINUTE_MS
    assert paths["buy_at_5"] == 1


def test_sequential_policy_waits_to_ten_then_skips_on_negative_q():
    frame = pd.DataFrame(
        [
            _row(0, base10=-1.0),
            _row(5, base10=-1.0),
            _row(10, base10=10.0),
        ]
    )
    models = SequentialModels(
        q10_buy=_q(-0.1),
        q5_buy=_q(-1.0),
        q5_wait=_q(1.0),
        q0_buy=_q(-1.0),
        q0_wait=_q(1.0),
    )

    trades, paths = select_sequential_policy(frame, models)

    assert trades.empty
    assert paths["skip_at_10"] == 1


def test_sequential_policy_missing_checkpoint_after_wait_skips():
    frame = pd.DataFrame(
        [
            _row(0, base10=-1.0),
            _row(6, base10=10.0),
        ]
    )
    models = SequentialModels(
        q10_buy=_q(1.0),
        q5_buy=_q(1.0),
        q5_wait=_q(1.0),
        q0_buy=_q(-1.0),
        q0_wait=_q(1.0),
    )

    trades, paths = select_sequential_policy(frame, models)

    assert trades.empty
    assert paths["skip_missing_checkpoint"] == 1
