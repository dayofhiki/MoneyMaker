import numpy as np
import pandas as pd

from victory_trader import state_calibrated_sequential_q as m
from victory_trader import state_fitted_sequential_q as base


def test_allowance_corrects_positive_selected_tail_bias():
    p = np.ones(10)
    y = np.zeros(10)
    correction, diagnostic = m.optimism_allowance(p, y, [str(i // 2) for i in range(10)])
    assert correction == 1.0
    assert diagnostic["calibration_days"] == 5


def test_allowance_never_inflates_predictions():
    correction, _ = m.optimism_allowance(np.ones(5), np.full(5, 2), list("abcde"))
    assert correction == 0.0


def test_insufficient_selected_days_disables_action():
    correction, diagnostic = m.optimism_allowance(np.ones(20), np.zeros(20), ["a"] * 20)
    assert np.isinf(correction)
    assert diagnostic["disabled_insufficient_days"]


def test_nonpositive_predictions_do_not_enter_bias_sample():
    correction, diagnostic = m.optimism_allowance(np.zeros(10), np.zeros(10), [str(i) for i in range(10)])
    assert np.isinf(correction)
    assert diagnostic["selected_rows"] == 0


def test_calibration_days_are_later_and_disjoint():
    frame = pd.DataFrame({"trading_day": [f"2026-01-{i:02d}" for i in range(1, 31)]})
    fit, cal = m.split_fit_calibration(frame)
    assert set(fit.trading_day).isdisjoint(set(cal.trading_day))
    assert max(fit.trading_day) < min(cal.trading_day)
    assert len(cal) == 6


def test_vectorized_checkpoints_preserve_reference_geometry():
    frame = pd.DataFrame({"trading_day": ["2026-01-02"] * 12, "ticker": ["AAA"] * 12,
                          "t": np.arange(12) * 60000, "c": [5.0] * 12,
                          "active_minute_fraction_15m": [1.0] * 12})
    expected = base.checkpoint_frames(frame)
    actual = m.checkpoints(frame)
    for offset in (0, 5, 10):
        pd.testing.assert_frame_equal(actual[offset], expected[offset])


def test_missing_checkpoint_is_zero_but_missing_label_stays_nan():
    src = pd.DataFrame({"trading_day": ["a"], "ticker": ["AAA"]})
    dest = pd.DataFrame({"trading_day": ["a", "a"], "ticker": ["AAA", "BBB"]})
    values = m.aligned_values(dest, src, [np.nan])
    assert np.isnan(values.iloc[0])
    assert values.iloc[1] == 0.0


def test_skip_wins_zero_tie_and_wait_wins_positive_tie():
    assert base._choose_mid_action(0.0, 0.0) == "SKIP"
    assert base._choose_mid_action(1.0, 1.0) == "WAIT"


def test_batch_policy_waits_to_exact_checkpoint_without_label_selection(monkeypatch):
    def fake_predict(frame, model):
        return frame[model].to_numpy(dtype=float)
    monkeypatch.setattr(m, "predict", fake_predict)
    row = {"trading_day": "a", "ticker": "AAA", "entry_price": 5.0,
           base.TARGET_COLUMN: 1.0, base.GROSS_COLUMN: 2.0, base.STRESS_COLUMN: -1.0}
    cp = {0: pd.DataFrame([{**row, "t": 0, "buy": -1.0, "wait": 1.0}]),
          5: pd.DataFrame([{**row, "t": 300000, "buy": 2.0, "wait": 1.0}]),
          10: pd.DataFrame([{**row, "t": 600000, "buy": 3.0, "wait": 0.0}])}
    models = {0: ("buy", "wait"), 5: ("buy", "wait"), 10: ("buy", None)}
    trades, paths = m.select_policy(cp, models)
    assert trades.iloc[0].t == 300000
    assert paths["buy_at_5"] == 1
    cp[5][base.TARGET_COLUMN] = np.nan
    trades, paths = m.select_policy(cp, models)
    assert trades.empty
    assert paths["buy_attempts"] == 1
    assert paths["unevaluated_attempts"] == 1
    assert paths["buy_at_10"] == 0


def test_wait_missing_checkpoint_terminates(monkeypatch):
    monkeypatch.setattr(m, "predict", lambda frame, model: frame[model].to_numpy(dtype=float))
    row = {"trading_day": "a", "ticker": "AAA", "buy": -1.0, "wait": 1.0}
    cp = {0: pd.DataFrame([row]), 5: pd.DataFrame(columns=row),
          10: pd.DataFrame([{**row, "buy": 5.0}])}
    models = {0: ("buy", "wait"), 5: ("buy", "wait"), 10: ("buy", None)}
    trades, paths = m.select_policy(cp, models)
    assert trades.empty
    assert paths["missing_checkpoint"] == 1
    assert paths["buy_attempts"] == 0
