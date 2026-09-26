from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.multi_event_entry_continuation import (
    EVENT_HORIZONS,
    attach_multi_event_entry_labels,
    recurrent_policy,
)


def _positions() -> pd.DataFrame:
    rows = []
    prices = [100, 101, 102, 103, 104, 105, 106, 107]
    for minute, price in enumerate(prices, start=1):
        rows.append(
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "minutes_held": float(minute),
                "state_t": minute * 60_000,
                "exit_reference_open": float(price),
                "log_current_close": np.log(float(price)),
            }
        )
    return pd.DataFrame(rows)


def _watch() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "minutes_since_hot": minute,
                "enter_3m_base_pct": float(minute),
            }
            for minute in range(1, 6)
        ]
    )


def test_multi_event_label_uses_fixed_observed_horizons() -> None:
    labeled = attach_multi_event_entry_labels(
        _watch(), _positions()
    )
    row = labeled.loc[
        labeled["minutes_since_hot"].eq(1)
    ].iloc[0]
    assert EVENT_HORIZONS == (1, 2, 3, 5)
    values = [
        row[f"entry_event{h}_base_pct"]
        for h in EVENT_HORIZONS
    ]
    assert all(np.isfinite(values))
    assert np.isclose(
        row["entry_multi_event_value_pct"],
        np.mean(values),
    )


def test_relative_target_preserves_abstain_option() -> None:
    labeled = attach_multi_event_entry_labels(
        _watch(), _positions()
    )
    first = labeled.loc[
        labeled["minutes_since_hot"].eq(1)
    ].iloc[0]
    second = labeled.loc[
        labeled["minutes_since_hot"].eq(2)
    ].iloc[0]
    expected = float(first["entry_multi_event_value_pct"]) - max(
        0.0, float(second["entry_multi_event_value_pct"])
    )
    assert np.isclose(
        first["enter_vs_wait_multi_event_advantage_pct"],
        expected,
    )


def test_recurrent_policy_waits_then_enters() -> None:
    frame = _watch()
    frame["predicted_multi_event_value_pct"] = [
        1.0, 1.2, 1.5, 1.4, 1.0
    ]
    frame["predicted_enter_vs_wait_advantage_pct"] = [
        -0.2, -0.1, 0.1, 0.1, 1.0
    ]
    trades, counts = recurrent_policy(frame)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_minute_after_hot"] == 3
    assert counts["wait_actions"] == 2


def test_recurrent_policy_abstains_at_minute5_if_value_nonpositive() -> None:
    frame = _watch()
    frame["predicted_multi_event_value_pct"] = [
        -1.0, -0.8, -0.5, -0.2, -0.1
    ]
    frame["predicted_enter_vs_wait_advantage_pct"] = [
        -0.1, -0.1, -0.1, -0.1, -0.1
    ]
    trades, counts = recurrent_policy(frame)
    assert trades.empty
    assert counts["abstained"] == 1
