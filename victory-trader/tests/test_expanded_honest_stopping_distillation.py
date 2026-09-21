from __future__ import annotations

import numpy as np
import pandas as pd

import victory_trader.expanded_honest_stopping_distillation as v41
from victory_trader.expanded_fitted_optimal_stopping import (
    _attach_exit_now_base,
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
    return build_continuation_rows(_state(), anchors)


def test_honest_target_uses_realized_later_teacher_policy(monkeypatch):
    rows = _rows().loc[
        lambda x: x["minutes_held"].isin([1, 2, 3])
    ].reset_index(drop=True)

    def fake_score(frame, _models):
        scored = _attach_exit_now_base(frame).reset_index(drop=True)
        scored["predicted_stopping_advantage_pct"] = scored[
            "minutes_held"
        ].map({1.0: 10.0, 2.0: -1.0, 3.0: -1.0})
        return scored

    monkeypatch.setattr(v41, "score_stopping_rows", fake_score)
    honest, _diag = v41.build_honest_teacher_targets(
        rows,
        teacher_models={},
        states={"2026-01": _state()},
    )

    minute2 = honest.loc[honest["minutes_held"].eq(2)].iloc[0]
    minute1 = honest.loc[honest["minutes_held"].eq(1)].iloc[0]

    # Teacher exits at minute 2, so forcing HOLD at minute 1 must inherit the
    # realized minute-2 EXIT value rather than a same-sample model prediction.
    assert np.isclose(
        minute1["honest_hold_downstream_value_pct"],
        minute2["exit_now_base_return_pct"],
    )
    assert np.isclose(
        minute1["honest_hold_advantage_pct"],
        minute2["exit_now_base_return_pct"]
        - minute1["exit_now_base_return_pct"],
    )
    assert np.isclose(
        minute1["teacher_realized_policy_value_pct"],
        minute2["exit_now_base_return_pct"],
    )


def test_honest_target_is_defined_even_when_teacher_would_exit_now(monkeypatch):
    rows = _rows().loc[
        lambda x: x["minutes_held"].isin([1, 2])
    ].reset_index(drop=True)

    def fake_score(frame, _models):
        scored = _attach_exit_now_base(frame).reset_index(drop=True)
        scored["predicted_stopping_advantage_pct"] = -1.0
        return scored

    monkeypatch.setattr(v41, "score_stopping_rows", fake_score)
    honest, _diag = v41.build_honest_teacher_targets(
        rows,
        teacher_models={},
        states={"2026-01": _state()},
    )

    minute1 = honest.loc[honest["minutes_held"].eq(1)].iloc[0]
    minute2 = honest.loc[honest["minutes_held"].eq(2)].iloc[0]

    # Student target asks the counterfactual "force HOLD once, then follow the
    # teacher", regardless of the teacher's current action.
    assert minute1["honest_hold_advantage_pct"] > 0
    assert np.isclose(
        minute1["honest_hold_downstream_value_pct"],
        minute2["teacher_realized_policy_value_pct"],
    )
    # The teacher itself exits immediately at minute 1.
    assert np.isclose(
        minute1["teacher_realized_policy_value_pct"],
        minute1["exit_now_base_return_pct"],
    )


def test_honest_target_never_uses_future_model_prediction_as_value(monkeypatch):
    rows = _rows().loc[
        lambda x: x["minutes_held"].isin([1, 2])
    ].reset_index(drop=True)

    def fake_score(frame, _models):
        scored = _attach_exit_now_base(frame).reset_index(drop=True)
        # Deliberately absurd predictions. They are action signals only and
        # must never become the economic downstream value.
        scored["predicted_stopping_advantage_pct"] = scored[
            "minutes_held"
        ].map({1.0: 999.0, 2.0: -999.0})
        return scored

    monkeypatch.setattr(v41, "score_stopping_rows", fake_score)
    honest, _diag = v41.build_honest_teacher_targets(
        rows,
        teacher_models={},
        states={"2026-01": _state()},
    )
    minute1 = honest.loc[honest["minutes_held"].eq(1)].iloc[0]

    assert abs(minute1["honest_hold_downstream_value_pct"]) < 100
    assert abs(minute1["honest_hold_advantage_pct"]) < 100
