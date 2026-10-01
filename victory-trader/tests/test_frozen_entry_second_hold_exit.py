import numpy as np
import pandas as pd
import pytest

from victory_trader.frozen_entry_second_hold_exit import (
    FEATURES, POLICIES, frozen_entries, position_states, replay, report,
)
from victory_trader.fresh_crack_entry_validation import (
    FROZEN_TURN_THRESHOLD, FROZEN_UTILITY_THRESHOLD,
)


def entry():
    return dict(trading_day="2026-05-05", ticker="TEST", hot_t=0,
                entry_t=0, entry_price=10.0, entry_setup_score_pct=4.0,
                entry_value_score_pct=1.0, entry_turn_score_pct=.5)


def seconds(prices=None, times=None):
    prices = prices or [10., 10.1, 10.2, 10.3, 10.4, 10.5]
    return pd.DataFrame(dict(t=times or list(range(0, len(prices) * 1000, 1000)),
                             o=prices, c=prices, h=prices, l=prices,
                             v=[100] * len(prices), n=[10] * len(prices)))


def test_completed_prefix_features_do_not_change_when_future_mutates():
    original = seconds()
    changed = original.copy()
    changed.loc[changed.t.ge(3000), ["o", "c", "h", "l"]] = 100.
    a = position_states(original, entry(), 5000)
    b = position_states(changed, entry(), 5000)
    pd.testing.assert_frame_equal(a.loc[a.decision_t.le(3000), list(FEATURES)],
                                  b.loc[b.decision_t.le(3000), list(FEATURES)])
    # The future execution label changes; it is deliberately not a feature.
    assert a.exit_price.iloc[2] != b.exit_price.iloc[2]


def test_stop_decision_executes_next_open_and_pays_gap():
    f = position_states(seconds([10., 9.8, 8., 8.1, 8.2]), entry(), 4000)
    outcome = replay(f, "terminal_60m")
    assert outcome["exit_reason"] == "hard_stop"
    assert outcome["exit_t"] == 2000
    assert outcome["gross_return_pct"] == pytest.approx(-20.)
    assert outcome["base_net_return_pct"] < -20.


def test_missing_execution_is_unresolved_not_backfilled():
    f = position_states(seconds([10., 9., 8.], [0, 1000, 10000]), entry(), 11000)
    outcome = replay(f, "terminal_60m")
    assert not outcome["resolved"]
    assert outcome["exit_reason"] == "unresolved_hard_stop"
    assert np.isnan(outcome["base_net_return_pct"])


def test_dynamic_hold_rechecks_and_exits_on_negative_advantage():
    f = position_states(seconds(), entry(), 5000)
    f["predicted_hold_300s_pct"] = [1., 1., -.1, 2., 2.]
    out = replay(f, "dynamic_300s")
    assert out["exit_reason"] == "model_exit"
    assert out["exit_t"] == 3000
    assert out["later_left_on_table_pp"] == pytest.approx(2.)


def test_frozen_entry_gates_no_strong_override_and_june_is_rejected():
    rows = pd.DataFrame([
        dict(trading_day="2026-05-05", held_out_day="2026-05-05", ticker="TEST", hot_t=0, decision_t=t,
             execution_t=t, execution_price=10., rich_setup_score_pct=4.,
             predicted_enter_utility_pct=100. if t == 1000 else FROZEN_UTILITY_THRESHOLD,
             predicted_turn_15s_pct=-10. if t == 1000 else FROZEN_TURN_THRESHOLD)
        for t in (1000, 2000)
    ])
    result, population = frozen_entries(rows)
    assert population == 1 and result.entry_t.iloc[0] == 2000
    with pytest.raises(ValueError, match="sealed"):
        frozen_entries(rows.assign(trading_day="2026-06-15"))


def test_terminal_missingness_and_unresolved_denominator_are_preserved():
    f = position_states(seconds([10., 10.1, 10.2]), entry(), 10000)
    assert not f.terminal_complete.any()
    rows = []
    for p in POLICIES:
        row = replay(f.assign(predicted_hold_60s_pct=1., predicted_hold_300s_pct=1.), p)
        rows.extend([row, {**row, "ticker": "MISSING", "resolved": False,
                          "base_net_return_pct": np.nan, "light_net_return_pct": np.nan,
                          "stress_net_return_pct": np.nan}])
    metrics = report(pd.DataFrame(rows), 3)
    assert metrics["immediate"]["resolution_rate"] == .5
    assert metrics["immediate"]["cash_adjusted_mean_base_net_pct"] is None
    assert metrics["terminal_60m"]["resolved"] == 0
