from datetime import date

import numpy as np
import pandas as pd

from victory_trader.short_interest_position_probe import PUBLICATION_SCHEDULE
from victory_trader.state_eight_k_enrichment import EIGHT_K_FEATURES
from victory_trader.state_multi_source_value import (
    EXTERNAL_FEATURES,
    multi_source_action_feature_frame,
    select_first_trades,
    success_check,
)
from victory_trader.state_short_interest_enrichment import (
    SHORT_INTEREST_FEATURES,
    _feature_row as short_interest_feature_row,
)
from victory_trader.state_short_volume_enrichment import SHORT_VOLUME_FEATURES


def _si_record(settlement, short_interest, adv, dtc):
    settlement_day = pd.Timestamp(settlement).date()
    return {
        "_settlement_day": settlement_day,
        "_publication_day": PUBLICATION_SCHEDULE[settlement_day],
        "settlement_date": settlement,
        "short_interest": short_interest,
        "avg_daily_volume": adv,
        "days_to_cover": dtc,
    }


def test_short_interest_state_uses_publication_date_strictly_before_day():
    records = [
        _si_record("2025-12-31", 1000, 500, 2.0),
        _si_record("2026-01-15", 1500, 600, 2.5),
    ]

    before_second_publication = short_interest_feature_row(
        records,
        date(2026, 1, 27),
    )
    assert before_second_publication["short_interest_latest"] == 1000.0
    assert before_second_publication["short_interest_reports_available"] == 1.0

    after_second_publication = short_interest_feature_row(
        records,
        date(2026, 1, 28),
    )
    assert after_second_publication["short_interest_latest"] == 1500.0
    assert after_second_publication["short_interest_change_pct"] == 50.0
    assert np.isclose(
        after_second_publication[
            "short_interest_avg_daily_volume_change_pct"
        ],
        20.0,
    )
    assert (
        after_second_publication["short_interest_days_to_cover_change"]
        == 0.5
    )


def test_multi_source_feature_frame_contains_all_frozen_external_features():
    frame = pd.DataFrame(
        {
            "return_from_previous_close_pct": [10.0],
            "minutes_since_10pct_cross": [1.0],
            "minutes_from_regular_open": [10.0],
            "active_minute_fraction_15m": [1.0],
            "c": [5.0],
            "previous_close": [4.0],
            **{column: [1.0] for column in SHORT_VOLUME_FEATURES},
            **{column: [2.0] for column in EIGHT_K_FEATURES},
            **{column: [3.0] for column in SHORT_INTEREST_FEATURES},
        }
    )
    features = multi_source_action_feature_frame(frame)

    assert len(EXTERNAL_FEATURES) == 29
    for column in SHORT_VOLUME_FEATURES:
        assert features.iloc[0][column] == 1.0
    for column in EIGHT_K_FEATURES:
        assert features.iloc[0][column] == 2.0
    for column in SHORT_INTEREST_FEATURES:
        assert features.iloc[0][column] == 3.0


def _trade_row(minute, *, ev, horizon=15, entry_price=5.0):
    row = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "predicted_best_base_ev_pct": 0.0,
        "predicted_best_horizon_min": horizon,
        "predicted_best_multi_source_ev_pct": ev,
        "predicted_best_multi_source_horizon_min": horizon,
    }
    for current in (5, 10, 15):
        row[f"buy_return_{current}m_base_net_return_pct"] = current / 10
        row[f"buy_return_{current}m_pct"] = current / 10 + 1.0
        row[f"buy_return_{current}m_stress_net_return_pct"] = current / 10 - 1.0
    return row


def test_multi_source_policy_uses_first_half_percent_signal():
    frame = pd.DataFrame(
        [
            _trade_row(0, ev=0.49),
            _trade_row(1, ev=0.50, horizon=10),
            _trade_row(2, ev=5.00, horizon=15),
        ]
    )
    trades = select_first_trades(frame, policy="multi_source_ev_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000
    assert int(trades.iloc[0]["action_horizon_min"]) == 10


def test_multi_source_first_signal_consumed_if_unfillable():
    frame = pd.DataFrame(
        [
            _trade_row(0, ev=1.0, entry_price=np.nan),
            _trade_row(1, ev=2.0),
        ]
    )
    trades = select_first_trades(frame, policy="multi_source_ev_cap1")
    assert trades.empty


def test_multi_source_success_check_requires_all_preregistered_rules():
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
                    "policy": "multi_source_ev_cap1",
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
            "short_volume_latest_coverage": [1.0, 1.0, 1.0],
            "short_interest_latest_coverage": [1.0, 1.0, 1.0],
            "eight_k_query_complete_coverage": [1.0, 1.0, 1.0],
        }
    )

    assert success_check(
        details,
        coverage,
        {"ci_low_pct": 0.1},
    )["all_pass"] is True

    details.loc[
        (details["month"].eq("2026-03"))
        & details["policy"].eq("multi_source_ev_cap1"),
        "day_balanced_base_mean_pct",
    ] = -0.1
    assert success_check(
        details,
        coverage,
        {"ci_low_pct": 0.1},
    )["all_pass"] is False
