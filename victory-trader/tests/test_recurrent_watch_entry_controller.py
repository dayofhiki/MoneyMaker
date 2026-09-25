import math

import numpy as np
import pandas as pd

from victory_trader.recurrent_watch_entry_controller import (
    add_action_targets,
    add_watch_path_features,
    run_controller,
)


EPISODE = {
    "trading_day": "2026-05-11",
    "ticker": "TEST",
    "hot_t": 1000,
}


def _row(minute, move, *, status="closed", value=-1.0):
    return {
        **EPISODE,
        "state_t": 1000 + minute * 60_000,
        "minutes_held": float(minute),
        "entry_to_current_close_pct": float(move),
        "drawdown_from_peak_pct": min(float(move), 0.0),
        "recovery_from_trough_pct": max(float(move), 0.0),
        "attention_score": 1.0 + minute / 10,
        "log_minute_volume": math.log1p(1000 + 100 * minute),
        "log_current_close": math.log(10.0 * (1.0 + move / 100.0)),
        "replay_status": status,
        "policy_net_return_pct": value,
    }


def test_watch_features_use_only_current_and_prior_states():
    frame = pd.DataFrame(
        [
            _row(1, 5.0),
            _row(2, 2.0),
            _row(3, 3.0),
        ]
    )
    result = add_watch_path_features(frame)
    minute2 = result.loc[result["minutes_held"].eq(2.0)].iloc[0]
    minute3 = result.loc[result["minutes_held"].eq(3.0)].iloc[0]

    assert minute2["watch_peak_age_minutes"] == 1.0
    assert minute2["watch_trough_age_minutes"] == 0.0
    assert math.isclose(
        minute2["watch_location_in_running_range"],
        2.0 / 5.0,
    )
    assert minute3["watch_peak_age_minutes"] == 2.0
    assert minute3["watch_trough_age_minutes"] == 1.0


def test_action_targets_use_best_strictly_later_closed_state():
    frame = pd.DataFrame(
        [
            _row(1, 1.0, value=-2.0),
            _row(2, 0.0, status="entry_unavailable", value=np.nan),
            _row(3, 2.0, value=4.0),
        ]
    )
    result = add_action_targets(frame).sort_values("minutes_held")
    assert result.iloc[0]["target_enter_now_pct"] == -2.0
    assert result.iloc[0]["target_wait_best_later_pct"] == 4.0
    assert pd.isna(result.iloc[1]["target_enter_now_pct"])
    assert result.iloc[1]["target_wait_best_later_pct"] == 4.0
    assert pd.isna(result.iloc[2]["target_wait_best_later_pct"])


def test_controller_waits_then_enters():
    frame = pd.DataFrame(
        [
            {
                **_row(1, 4.0, value=-1.0),
                "path_pred_enter_now_pct": -0.5,
                "path_pred_wait_pct": 2.0,
            },
            {
                **_row(2, 1.0, value=3.0),
                "path_pred_enter_now_pct": 1.5,
                "path_pred_wait_pct": 0.5,
            },
        ]
    )
    selected, audit = run_controller(frame, prefix="path")
    assert len(selected) == 1
    assert selected.iloc[0]["minutes_held"] == 2.0
    assert audit["action_counts"]["WAIT"] == 1
    assert audit["entered_episodes"] == 1


def test_controller_can_enter_immediately_or_drop():
    enter = pd.DataFrame(
        [
            {
                **_row(1, 1.0, value=2.0),
                "path_pred_enter_now_pct": 2.0,
                "path_pred_wait_pct": 1.0,
            },
            {
                **_row(2, 2.0, value=1.0),
                "path_pred_enter_now_pct": 1.0,
                "path_pred_wait_pct": np.nan,
            },
        ]
    )
    selected, _ = run_controller(enter, prefix="path")
    assert selected.iloc[0]["minutes_held"] == 1.0

    drop = enter.copy()
    drop.loc[drop.index[0], "path_pred_enter_now_pct"] = -1.0
    drop.loc[drop.index[0], "path_pred_wait_pct"] = -0.5
    selected, audit = run_controller(drop, prefix="path")
    assert selected.empty
    assert audit["dropped_episodes"] == 1


def test_unavailable_enter_retries_later_state():
    frame = pd.DataFrame(
        [
            {
                **_row(
                    1,
                    1.0,
                    status="entry_unavailable",
                    value=np.nan,
                ),
                "path_pred_enter_now_pct": 2.0,
                "path_pred_wait_pct": 1.0,
            },
            {
                **_row(2, 0.5, value=3.0),
                "path_pred_enter_now_pct": 1.0,
                "path_pred_wait_pct": np.nan,
            },
        ]
    )
    selected, audit = run_controller(frame, prefix="path")
    assert len(selected) == 1
    assert selected.iloc[0]["minutes_held"] == 2.0
    assert audit["entry_unavailable_retries"] == 1
