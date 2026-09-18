import numpy as np
import pandas as pd

from victory_trader.state_pairwise_buy_wait import (
    pairwise_label,
    select_policy_trades,
    success_check,
)


MINUTE_MS = 60_000


def _row(
    minute,
    *,
    base10,
    prob=0.5,
    day="2026-01-02",
    ticker="AAA",
):
    return {
        "trading_day": day,
        "ticker": ticker,
        "t": minute * MINUTE_MS,
        "c": 5.0,
        "active_minute_fraction_15m": 1.0,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": 5.0,
        "buy_return_10m_base_net_return_pct": base10,
        "buy_return_10m_pct": base10 + 1.0,
        "buy_return_10m_stress_net_return_pct": base10 - 1.0,
        "predicted_now_beats_wait5_prob": prob,
    }


def test_pairwise_label_compares_exact_five_minute_future_only():
    frame = pd.DataFrame(
        [
            _row(0, base10=1.0),
            _row(4, base10=100.0),
            _row(5, base10=2.0),
            _row(10, base10=-1.0),
        ]
    )
    label = pairwise_label(frame)

    assert label.iloc[0] == 0.0
    assert np.isnan(label.iloc[1])
    assert label.iloc[2] == 1.0
    assert np.isnan(label.iloc[3])


def test_pairwise_policy_waits_then_buys_at_exact_checkpoint():
    frame = pd.DataFrame(
        [
            _row(0, base10=-1.0, prob=0.40),
            _row(5, base10=3.0, prob=0.60),
            _row(10, base10=5.0, prob=0.90),
        ]
    )
    trades = select_policy_trades(frame, policy="pairwise_wait_cap1")

    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 5 * MINUTE_MS
    assert trades.iloc[0]["realized_base_net_return_pct"] == 3.0


def test_pairwise_policy_all_three_waits_skips_episode():
    frame = pd.DataFrame(
        [
            _row(0, base10=1.0, prob=0.49),
            _row(5, base10=2.0, prob=0.49),
            _row(10, base10=3.0, prob=0.49),
        ]
    )
    trades = select_policy_trades(frame, policy="pairwise_wait_cap1")
    assert trades.empty


def test_pairwise_policy_missing_required_checkpoint_ends_path():
    frame = pd.DataFrame(
        [
            _row(0, base10=1.0, prob=0.40),
            _row(6, base10=10.0, prob=0.90),
            _row(10, base10=10.0, prob=0.90),
        ]
    )
    trades = select_policy_trades(frame, policy="pairwise_wait_cap1")
    assert trades.empty


def test_fixed_wait_baselines_use_exact_offsets():
    frame = pd.DataFrame(
        [
            _row(0, base10=1.0, prob=0.5),
            _row(5, base10=2.0, prob=0.5),
            _row(10, base10=3.0, prob=0.5),
        ]
    )
    earliest = select_policy_trades(
        frame, policy="earliest_eligible_10m_cap1"
    )
    wait5 = select_policy_trades(frame, policy="fixed_wait5_10m_cap1")
    wait10 = select_policy_trades(frame, policy="fixed_wait10_10m_cap1")

    assert earliest.iloc[0]["realized_base_net_return_pct"] == 1.0
    assert wait5.iloc[0]["realized_base_net_return_pct"] == 2.0
    assert wait10.iloc[0]["realized_base_net_return_pct"] == 3.0


def test_pairwise_success_check_requires_all_rules():
    details_rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        details_rows.extend(
            [
                {
                    "month": month,
                    "policy": "earliest_eligible_10m_cap1",
                    "trades": 30,
                    "base_mean_pct": 0.5,
                    "day_balanced_base_mean_pct": 0.5,
                    "base_p05_pct": -10.0,
                    "stress_mean_pct": -1.0,
                },
                {
                    "month": month,
                    "policy": "pairwise_wait_cap1",
                    "trades": 20,
                    "base_mean_pct": 1.0,
                    "day_balanced_base_mean_pct": 1.0,
                    "base_p05_pct": -9.0,
                    "stress_mean_pct": 0.0,
                },
            ]
        )
    details = pd.DataFrame(details_rows)
    diagnostics = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02", "2026-03"],
            "roc_auc": [0.51, 0.52, 0.53],
        }
    )
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
        diagnostics,
        coverage,
        {"ci_low_pct": 0.1},
    )["all_pass"] is True

    diagnostics.loc[2, "roc_auc"] = 0.49
    assert success_check(
        details,
        diagnostics,
        coverage,
        {"ci_low_pct": 0.1},
    )["all_pass"] is False
