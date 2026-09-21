from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.expanded_dual_horizon_patience_override import (
    _decision_sources,
    build_v38_patience_targets,
    override_feature_frame,
    split_calibration_days,
)
from victory_trader.expanded_path_aware_continuation import (
    MINUTE_MS,
    build_continuation_rows,
)


def _state(last_minute: int = 35) -> pd.DataFrame:
    rows = []
    for minute in range(last_minute + 1):
        o = 10.0 + 0.1 * minute
        rows.append(
            {
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "t": minute * MINUTE_MS,
                "o": o,
                "h": o + 0.05,
                "l": o - 0.05,
                "c": o + 0.02,
                "previous_close": 9.5,
                "trailing_return_1m_pct": float(minute) * 0.1,
                "hod_distance_pct": -1.0,
                "regular_vwap_distance_pct": 1.0,
                "volume_accel_1m_vs_prior20m": 1.0,
                "bar_range_pct": 1.0,
                "bar_close_location": 0.5,
                "phase": "impulse",
            }
        )
    return pd.DataFrame(rows)


def _rows() -> pd.DataFrame:
    anchors = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "t": 0,
                "opportunity_probability": 0.8,
            }
        ]
    )
    rows = build_continuation_rows(_state(), anchors)
    rows["predicted_excess_option_value_pct"] = 1.0
    return rows


def test_calibration_split_is_chronological_and_disjoint():
    frame = pd.DataFrame(
        {
            "trading_day": [
                f"2026-01-{day:02d}" for day in range(1, 13)
            ],
            "x": range(12),
        }
    )
    a, b, a_days, b_days = split_calibration_days(frame)
    assert a_days == [f"2026-01-{day:02d}" for day in range(1, 7)]
    assert b_days == [f"2026-01-{day:02d}" for day in range(7, 13)]
    assert set(a["trading_day"]).isdisjoint(set(b["trading_day"]))


def test_patience_target_only_exists_when_v38_teacher_would_exit():
    rows = _rows().loc[
        lambda x: x["minutes_held"].isin([1, 2, 3])
    ].reset_index(drop=True)
    rows["short_hold_probability"] = rows["minutes_held"].map(
        {1.0: 0.4, 2.0: 0.8, 3.0: 0.4}
    )

    honest, _ = build_v38_patience_targets(
        rows,
        {"2026-01": _state()},
    )
    minute1 = honest.loc[honest["minutes_held"].eq(1)].iloc[0]
    minute2 = honest.loc[honest["minutes_held"].eq(2)].iloc[0]

    assert pd.notna(minute1["patience_override_advantage_pct"])
    assert np.isnan(minute2["patience_override_advantage_pct"])


def test_patience_target_is_force_hold_once_then_follow_teacher():
    rows = _rows().loc[
        lambda x: x["minutes_held"].isin([1, 2, 3])
    ].reset_index(drop=True)
    # Exit at minute 1, HOLD at minute 2, EXIT at minute 3.
    rows["short_hold_probability"] = rows["minutes_held"].map(
        {1.0: 0.4, 2.0: 0.8, 3.0: 0.4}
    )

    honest, _ = build_v38_patience_targets(
        rows,
        {"2026-01": _state()},
    )
    m1 = honest.loc[honest["minutes_held"].eq(1)].iloc[0]
    m2 = honest.loc[honest["minutes_held"].eq(2)].iloc[0]
    m3 = honest.loc[honest["minutes_held"].eq(3)].iloc[0]

    # Teacher HOLD at minute 2 must inherit the realized minute-3 teacher
    # value, and the minute-1 override target must inherit that same value.
    assert np.isclose(
        m2["v38_teacher_realized_value_pct"],
        m3["v38_teacher_realized_value_pct"],
    )
    assert np.isclose(
        m1["forced_one_minute_value_pct"],
        m2["v38_teacher_realized_value_pct"],
    )
    assert np.isclose(
        m1["patience_override_advantage_pct"],
        m1["forced_one_minute_value_pct"]
        - m1["exit_now_base_return_pct"],
    )


def test_override_model_inputs_are_exactly_two_frozen_scores():
    frame = pd.DataFrame(
        {
            "short_hold_probability": [0.2, 0.7],
            "predicted_excess_option_value_pct": [1.1, -0.3],
            "unrelated_future_feature": [999.0, -999.0],
        }
    )
    features = override_feature_frame(frame)
    assert features.columns.tolist() == [
        "short_hold_probability",
        "predicted_excess_option_value_pct",
    ]


def test_decision_source_distinguishes_short_and_patience_holds():
    decisions = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "minutes_held": 1,
                "action": "HOLD",
            },
            {
                "trading_day": "2026-01-02",
                "ticker": "BBB",
                "minutes_held": 1,
                "action": "HOLD",
            },
            {
                "trading_day": "2026-01-02",
                "ticker": "CCC",
                "minutes_held": 1,
                "action": "EXIT",
            },
        ]
    )
    scored = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "minutes_held": 1,
                "short_hold_probability": 0.7,
                "predicted_excess_option_value_pct": -1.0,
                "predicted_patience_override_pct": -1.0,
            },
            {
                "trading_day": "2026-01-02",
                "ticker": "BBB",
                "minutes_held": 1,
                "short_hold_probability": 0.4,
                "predicted_excess_option_value_pct": 1.0,
                "predicted_patience_override_pct": 0.2,
            },
            {
                "trading_day": "2026-01-02",
                "ticker": "CCC",
                "minutes_held": 1,
                "short_hold_probability": 0.4,
                "predicted_excess_option_value_pct": -0.1,
                "predicted_patience_override_pct": -0.2,
            },
        ]
    )
    tagged = _decision_sources(decisions, scored)
    assert tagged["decision_source"].tolist() == [
        "short_term_hold",
        "patience_override_hold",
        "dual_horizon_exit",
    ]
