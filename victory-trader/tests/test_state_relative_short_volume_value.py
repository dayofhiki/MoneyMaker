import numpy as np
import pandas as pd

from victory_trader.state_relative_short_volume_value import (
    relative_short_volume_action_feature_frame,
    select_first_trades,
    success_check,
)
from victory_trader.state_short_volume_context import (
    RELATIVE_SHORT_VOLUME_FEATURES,
)
from victory_trader.state_short_volume_enrichment import SHORT_VOLUME_FEATURES


def test_relative_feature_frame_adds_absolute_and_context_features():
    frame = pd.DataFrame(
        {
            "return_from_previous_close_pct": [10.0],
            "minutes_since_10pct_cross": [1.0],
            "minutes_from_regular_open": [10.0],
            "active_minute_fraction_15m": [1.0],
            "c": [5.0],
            "previous_close": [4.0],
            **{column: [1.0] for column in SHORT_VOLUME_FEATURES},
            **{column: [2.0] for column in RELATIVE_SHORT_VOLUME_FEATURES},
        }
    )
    features = relative_short_volume_action_feature_frame(frame)

    for column in SHORT_VOLUME_FEATURES:
        assert column in features.columns
        assert features.iloc[0][column] == 1.0
    for column in RELATIVE_SHORT_VOLUME_FEATURES:
        assert column in features.columns
        assert features.iloc[0][column] == 2.0


def _row(minute, *, relative_ev, horizon=15, entry_price=5.0):
    row = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "predicted_best_base_ev_pct": 0.0,
        "predicted_best_horizon_min": horizon,
        "predicted_best_short_volume_ev_pct": 0.0,
        "predicted_best_short_volume_ev_horizon_min": horizon,
        "predicted_best_relative_short_volume_ev_pct": relative_ev,
        "predicted_best_relative_short_volume_ev_horizon_min": horizon,
    }
    for current in (5, 10, 15):
        row[f"buy_return_{current}m_base_net_return_pct"] = current / 10
        row[f"buy_return_{current}m_pct"] = current / 10 + 1.0
        row[f"buy_return_{current}m_stress_net_return_pct"] = current / 10 - 1.0
    return row


def test_relative_policy_enters_first_half_percent_signal():
    frame = pd.DataFrame(
        [
            _row(0, relative_ev=0.49),
            _row(1, relative_ev=0.50, horizon=10),
            _row(2, relative_ev=2.00, horizon=15),
        ]
    )
    trades = select_first_trades(
        frame,
        policy="relative_short_volume_ev_cap1",
    )
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000
    assert int(trades.iloc[0]["action_horizon_min"]) == 10


def test_relative_first_signal_is_consumed_when_unfillable():
    frame = pd.DataFrame(
        [
            _row(0, relative_ev=0.75, entry_price=np.nan),
            _row(1, relative_ev=1.50),
        ]
    )
    trades = select_first_trades(
        frame,
        policy="relative_short_volume_ev_cap1",
    )
    assert trades.empty


def test_relative_success_check_requires_all_preregistered_rules():
    rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        rows.extend(
            [
                {
                    "month": month,
                    "policy": "earliest_ev_cap1",
                    "trades": 20,
                    "base_mean_pct": 0.5,
                    "day_balanced_base_mean_pct": 0.5,
                    "base_p05_pct": -10.0,
                    "stress_mean_pct": -1.0,
                },
                {
                    "month": month,
                    "policy": "relative_short_volume_ev_cap1",
                    "trades": 20,
                    "base_mean_pct": 1.0,
                    "day_balanced_base_mean_pct": 1.0,
                    "base_p05_pct": -9.0,
                    "stress_mean_pct": 0.0,
                },
            ]
        )
    details = pd.DataFrame(rows)
    coverage = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02", "2026-03"],
            "relative_latest_percentile_coverage": [1.0, 1.0, 1.0],
            "relative_latest_difference_coverage": [1.0, 1.0, 1.0],
        }
    )
    bootstrap = {"ci_low_pct": 0.1}
    assert success_check(details, coverage, bootstrap)["all_pass"] is True

    details.loc[
        (details["month"].eq("2026-03"))
        & details["policy"].eq("relative_short_volume_ev_cap1"),
        "day_balanced_base_mean_pct",
    ] = -0.1
    assert success_check(details, coverage, bootstrap)["all_pass"] is False
