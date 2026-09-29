from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.multi_event_option_value import (
    OPTION_HORIZON_EVENTS,
    attach_option_rank_target,
    attach_option_value_target,
)


def _trajectory() -> pd.DataFrame:
    rows = []
    returns = [0.0, -0.2, 0.5, 1.0, 0.7, 1.8, 1.2]
    for i, value in enumerate(returns):
        rows.append(
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "state_t": (i + 1) * 60_000,
                "minutes_held": float(i + 1),
                "exit_now_base_return_pct": value,
            }
        )
    return pd.DataFrame(rows)


def test_option_value_uses_best_of_next_five_events() -> None:
    enriched = attach_option_value_target(_trajectory())
    first = enriched.iloc[0]
    assert OPTION_HORIZON_EVENTS == 5
    assert np.isclose(first["option_best_future_exit_pct"], 1.8)
    assert np.isclose(first["option_value_5event_pct"], 1.8)


def test_option_target_requires_full_five_event_horizon() -> None:
    enriched = attach_option_value_target(_trajectory())
    assert np.isfinite(enriched.iloc[0]["option_value_5event_pct"])
    assert np.isfinite(enriched.iloc[1]["option_value_5event_pct"])
    assert pd.isna(enriched.iloc[2]["option_value_5event_pct"])


def test_future_change_cannot_change_earlier_features_but_changes_label() -> None:
    base = attach_option_value_target(_trajectory())
    changed_frame = _trajectory()
    changed_frame.loc[5, "exit_now_base_return_pct"] = 9.0
    changed = attach_option_value_target(changed_frame)

    assert not np.isclose(
        base.iloc[0]["option_value_5event_pct"],
        changed.iloc[0]["option_value_5event_pct"],
    )
    assert base.iloc[0]["state_t"] == changed.iloc[0]["state_t"]
    assert base.iloc[0]["minutes_held"] == changed.iloc[0]["minutes_held"]


def test_option_rank_is_within_day_and_monotone() -> None:
    frame = pd.DataFrame(
        {
            "trading_day": ["A", "A", "A", "B", "B"],
            "option_value_5event_pct": [0.0, 1.0, 2.0, 10.0, 20.0],
        }
    )
    ranked = attach_option_rank_target(frame)
    a = ranked.loc[
        ranked["trading_day"].eq("A"), "option_value_day_rank"
    ].to_numpy()
    b = ranked.loc[
        ranked["trading_day"].eq("B"), "option_value_day_rank"
    ].to_numpy()
    assert np.all(np.diff(a) > 0)
    assert np.all(np.diff(b) > 0)
    assert np.isclose(a[-1], 1.0)
    assert np.isclose(b[-1], 1.0)


def test_thirty_minute_cap_censors_target() -> None:
    frame = _trajectory()
    frame.loc[5, "minutes_held"] = 31.0
    enriched = attach_option_value_target(frame)
    assert pd.isna(enriched.iloc[0]["option_value_5event_pct"])
