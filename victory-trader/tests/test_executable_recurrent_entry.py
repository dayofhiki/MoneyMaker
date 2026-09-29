import numpy as np
import pandas as pd
import pytest

from victory_trader.executable_recurrent_entry import decide, settle, development_gate


def states(evs, advantages=None):
    return pd.DataFrame([
        dict(trading_day="2026-06-23", ticker="TEST", hot_t=100,
             minutes_since_hot=i + 1, predicted_entry_ev_pct=ev,
             predicted_relative_advantage_pct=(advantages or [1] * len(evs))[i],
             enter_3m_base_pct=2.0)
        for i, ev in enumerate(evs)
    ])


def test_wait_reobserve_then_enter():
    frame = states([-1, 1, 2, 3, 4], [1, -1, 1, 1, 1])
    decision = decide(frame).iloc[0]
    assert decision.action == "ENTER"
    assert decision.entry_minute_after_hot == 3
    assert decision.wait_actions == 2
    assert decide(frame, reobserve=False).iloc[0].action == "ABSTAIN"


def test_no_forced_terminal_purchase():
    assert decide(states([-1] * 5)).iloc[0].action == "ABSTAIN"
    assert decide(states([0] * 5)).iloc[0].action == "ABSTAIN"
    assert decide(states([-1] * 4 + [1], [-1] * 5)).iloc[0].entry_minute_after_hot == 5


def test_future_outcomes_cannot_change_actions():
    frame = states([-1, 1, 1, 1, 1])
    expected = decide(frame)
    frame["enter_3m_base_pct"] = [9999, np.nan, -9999, 0, 500]
    frame["oracle_state_value_pct"] = 1e10
    pd.testing.assert_frame_equal(expected, decide(frame))
    resolved = settle(expected, frame).iloc[0]
    assert resolved.action == "ENTER"
    assert not resolved.resolved
    assert pd.isna(resolved.realized_base_return_pct)


def test_future_checkpoints_cannot_change_prior_purchase():
    frame = states([1] * 5)
    expected = decide(frame)
    frame.loc[1:, "predicted_entry_ev_pct"] = -1000
    pd.testing.assert_frame_equal(expected, decide(frame))


def test_missing_checkpoint_not_silently_abstained_or_skipped():
    frame = states([-1, -1, 1, 1, 1]).drop(index=1)
    decision = decide(frame)
    assert decision.iloc[0].action == "COVERAGE_MISS"
    assert not settle(decision, frame).iloc[0].resolved


def test_duplicate_checkpoint_rejected():
    frame = states([1])
    with pytest.raises(ValueError, match="duplicate"):
        decide(pd.concat([frame, frame]))


def test_all_cash_cannot_pass_profitability_gate():
    report = {"fully_resolved": True, "resolved_trade_metrics": {"trades": 0}}
    assert not development_gate(report, {"ci_low_pct": 1})


def test_unresolved_and_negative_trades_cannot_pass_gate():
    metrics = {"trades": 100, "day_balanced_mean_pct": -0.1, "by_day": {str(i): 1 for i in range(5)}}
    report = {"fully_resolved": True, "resolved_trade_metrics": metrics}
    assert not development_gate(report, {"ci_low_pct": 1})
    metrics["day_balanced_mean_pct"] = 1
    report["fully_resolved"] = False
    assert not development_gate(report, {"ci_low_pct": 1})
