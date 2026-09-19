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


def test_one_day_does_not_produce_degenerate_confidence_interval(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("one-day sample must not be bootstrapped")
    monkeypatch.setattr(m, "day_cluster_bootstrap", forbidden)
    trades = pd.DataFrame({"policy": [m.POLICY], "trading_day": ["a"],
                           "realized_base_net_return_pct": [-3.8]})
    result = m.policy_bootstrap(trades)
    assert result["days"] == 1
    assert not result["bootstrap_available"]
    assert np.isnan(result["ci_low_pct"])
    assert result["day_balanced_mean_pct"] == -3.8


def test_zero_evaluated_days_have_no_bootstrap():
    assert not m.policy_bootstrap(pd.DataFrame())["bootstrap_available"]
    trades = pd.DataFrame({"policy": [m.POLICY], "trading_day": ["a"],
                           "realized_base_net_return_pct": [np.nan]})
    assert m.policy_bootstrap(trades)["days"] == 0


def test_two_day_sample_preserves_existing_bootstrap_calculation():
    trades = pd.DataFrame({"policy": [m.POLICY, m.POLICY], "trading_day": ["a", "b"],
                           "realized_base_net_return_pct": [-1.0, 1.0]})
    expected = m.day_cluster_bootstrap(trades, policy=m.POLICY, samples=10000)
    assert m.policy_bootstrap(trades) == {**expected, "bootstrap_available": True}


def test_attempt_reasons_keep_overlapping_missing_flags():
    row = pd.Series({"entry_price": np.nan, base.GROSS_COLUMN: np.nan})
    d = m.attempt_diagnostic(row, 0, None)
    assert d["evaluation_reason"] == "entry_missing"
    assert d["gross_missing"] and d["base_missing"] and not d["legacy_evaluated"]


def test_valid_entry_missing_exit_is_not_called_unfilled():
    row = pd.Series({"entry_price": 5.0, base.GROSS_COLUMN: np.nan,
                     base.TARGET_COLUMN: np.nan, base.STRESS_COLUMN: np.nan})
    d = m.attempt_diagnostic(row, 5, None)
    assert d["evaluation_reason"] == "gross_label_missing"
    assert not d["entry_missing"]


def test_attempt_reason_scenario_missing_and_nonfinite():
    row = pd.Series({"entry_price": 5.0, base.GROSS_COLUMN: 1.0,
                     base.TARGET_COLUMN: np.nan, base.STRESS_COLUMN: 0.0})
    assert m.attempt_diagnostic(row, 0, None)["evaluation_reason"] == "scenario_label_missing"
    row[base.TARGET_COLUMN] = np.inf
    assert m.attempt_diagnostic(row, 0, {})["evaluation_reason"] == "return_nonfinite"
    row["entry_price"] = 0
    assert m.attempt_diagnostic(row, 0, None)["evaluation_reason"] == "entry_invalid"


def test_audit_never_changes_attempts_or_falls_through(monkeypatch):
    monkeypatch.setattr(m, "predict", lambda frame, model: frame[model].to_numpy(dtype=float))
    row = {"trading_day": "a", "ticker": "AAA", "t": 0, "entry_price": 5.0,
           base.GROSS_COLUMN: np.nan, base.TARGET_COLUMN: np.nan,
           base.STRESS_COLUMN: np.nan, "buy": 2.0, "wait": 0.0}
    cp = {stage: pd.DataFrame([row]) for stage in (0, 5, 10)}
    models = {0: ("buy", "wait"), 5: ("buy", "wait"), 10: ("buy", None)}
    original, counts = m.select_policy(cp, models)
    ledger = []
    audited, audited_counts = m.select_policy(cp, models, attempt_records=ledger)
    pd.testing.assert_frame_equal(original, audited)
    assert counts == audited_counts
    assert len(ledger) == 1 and ledger[0]["evaluation_reason"] == "gross_label_missing"


def test_forward_split_uses_only_past_months():
    monthly = {m: pd.DataFrame({"trading_day": [m + "-15"]})
               for m in ("2026-01", "2026-02", "2026-03")}
    folds = list(m.evaluation_folds(monthly, "forward"))
    assert len(folds) == 1
    month, training, _ = folds[0]
    assert month == "2026-03" and training == ["2026-01", "2026-02"]
    assert len(list(m.evaluation_folds(monthly, "lomo"))) == 3


def test_forward_rejects_mislabelled_future_dates():
    import pytest
    monthly = {m: pd.DataFrame({"trading_day": ["2026-03-15"]})
               for m in ("2026-01", "2026-02", "2026-03")}
    with pytest.raises(ValueError, match="overlaps"):
        list(m.evaluation_folds(monthly, "forward"))


def test_forward_month_bounds_keep_older_months_as_training_only():
    monthly = {
        month: pd.DataFrame({"trading_day": [month + "-15"]})
        for month in (
            "2025-09",
            "2025-10",
            "2025-11",
            "2025-12",
            "2026-01",
            "2026-02",
            "2026-03",
        )
    }
    folds = list(
        m.evaluation_folds(
            monthly,
            "forward",
            evaluation_min_month="2026-01",
            evaluation_max_month="2026-03",
        )
    )
    assert [month for month, _, _ in folds] == [
        "2026-01",
        "2026-02",
        "2026-03",
    ]
    assert folds[0][1] == ["2025-09", "2025-10", "2025-11", "2025-12"]
    assert folds[1][1][-1] == "2026-01"
    assert folds[2][1][-1] == "2026-02"
