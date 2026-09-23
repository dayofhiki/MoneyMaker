from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.second_path_attention_probe import (
    BASELINE_FEATURES,
    SECOND_FEATURES,
    _annotate_scan,
    _second_frame,
    fit_and_evaluate,
    second_feature_window,
    second_path_features,
)


def test_second_path_features_use_only_completed_seconds():
    decision_t = 60_000
    seconds = pd.DataFrame(
        [
            {"t": 49_000, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0, "v": 10, "n": 1},
            {"t": 50_000, "o": 10.0, "h": 10.1, "l": 10.0, "c": 10.1, "v": 20, "n": 2},
            {"t": 59_000, "o": 10.1, "h": 10.2, "l": 10.1, "c": 10.2, "v": 30, "n": 3},
            # This bar begins exactly at the decision and is not yet complete.
            {"t": 60_000, "o": 99.0, "h": 99.0, "l": 99.0, "c": 99.0, "v": 999, "n": 99},
        ]
    )

    features = second_path_features(seconds, decision_t)

    assert features["active_seconds_60"] == 3.0
    assert features["active_seconds_10"] == 2.0
    assert features["last_activity_age_s"] == 0.0
    assert features["sec_volume_last10_share"] == 50.0 / 60.0


def test_fit_and_evaluate_keeps_last_two_sessions_untouched():
    rng = np.random.default_rng(17)
    rows = []
    days = [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
    ]
    for day_index, day in enumerate(days):
        for i in range(50):
            baseline_signal = i / 50.0
            second_signal = ((i * 7 + day_index) % 50) / 50.0
            row = {
                "trading_day": day,
                "ticker": f"T{i % 10:02d}",
                "future_max_3m_return_pct": (
                    baseline_signal + 2.0 * second_signal + rng.normal(0, 0.02)
                ),
            }
            for feature in BASELINE_FEATURES:
                row[feature] = baseline_signal
            for feature in SECOND_FEATURES:
                row[feature] = second_signal
            rows.append(row)

    modeled, summary = fit_and_evaluate(pd.DataFrame(rows))

    assert summary["fit_days"] == days[:4]
    assert summary["eval_days"] == days[4:]
    assert summary["fit_rows"] == 200
    assert summary["eval_rows"] == 100
    assert set(modeled.loc[modeled["split"].eq("eval"), "trading_day"]) == set(days[4:])
    assert modeled["baseline_prediction"].notna().all()
    assert modeled["extended_prediction"].notna().all()


def test_second_feature_window_matches_original_boolean_boundaries():
    decision_t = 60_000
    seconds = pd.DataFrame(
        [
            {"t": 0, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1, "n": 1},
            {"t": 1_000, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1, "n": 1},
            {"t": 58_000, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1, "n": 1},
            {"t": 59_000, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1, "n": 1},
            {"t": 60_000, "o": 9.0, "h": 9.0, "l": 9.0, "c": 9.0, "v": 9, "n": 9},
        ]
    )
    expected = seconds.loc[
        seconds["t"].ge(decision_t - 60_000)
        & (seconds["t"] + 1_000).le(decision_t)
    ]
    actual = second_feature_window(seconds, decision_t)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_annotate_scan_resets_lagged_features_at_trading_day_boundary():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-14",
                "ticker": "AAA",
                "t": 60_000,
                "o": 1.0,
                "h": 1.2,
                "l": 1.0,
                "c": 1.2,
                "v": 100.0,
                "n": 10.0,
                "return_from_previous_close_pct": 20.0,
            },
            {
                "trading_day": "2026-05-15",
                "ticker": "AAA",
                "t": 60_000,
                "o": 1.0,
                "h": 1.01,
                "l": 0.99,
                "c": 1.0,
                "v": 10.0,
                "n": 2.0,
                "return_from_previous_close_pct": 0.0,
            },
        ]
    )

    result = _annotate_scan(scan)
    second_day = result.loc[result["trading_day"].eq("2026-05-15")].iloc[0]

    assert pd.isna(second_day["minute_return_1m_pct"])
    assert pd.isna(second_day["return_accel_1m_pct"])
    assert pd.isna(second_day["volume_ratio_prev1"])
    assert pd.isna(second_day["transactions_ratio_prev1"])
    assert bool(second_day["already_runner"]) is False


def test_second_frame_enforces_searchsorted_ordering_invariant():
    frame = _second_frame(
        {
            "results": [
                {"t": 2_000, "o": 2.0, "h": 2.0, "l": 2.0, "c": 2.0, "v": 2.0},
                {"t": 1_000, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0},
                {"t": 2_000, "o": 2.1, "h": 2.1, "l": 2.1, "c": 2.1, "v": 3.0},
            ]
        }
    )

    assert frame["t"].is_monotonic_increasing
    assert frame["t"].tolist() == [1_000, 2_000]
    assert frame.iloc[-1]["c"] == 2.1
