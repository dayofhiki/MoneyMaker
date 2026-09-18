import pandas as pd

from victory_trader.state_buy_wait_skip import (
    ACTION_BUY,
    ACTION_SKIP,
    ACTION_WAIT,
    select_policy,
    success_check,
    three_action_label,
)


MINUTE_MS = 60_000


def _row(minute, *, base10, action=ACTION_BUY, day="2026-01-02"):
    probs = {
        ACTION_BUY: (0.8, 0.1, 0.1),
        ACTION_WAIT: (0.1, 0.8, 0.1),
        ACTION_SKIP: (0.1, 0.1, 0.8),
    }[action]
    return {
        "trading_day": day,
        "ticker": "AAA",
        "t": minute * MINUTE_MS,
        "c": 5.0,
        "active_minute_fraction_15m": 1.0,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": 5.0,
        "buy_return_10m_base_net_return_pct": base10,
        "buy_return_10m_pct": base10 + 1.0,
        "buy_return_10m_stress_net_return_pct": base10 - 1.0,
        "predicted_policy_action": action,
        "predicted_action_prob_buy": probs[0],
        "predicted_action_prob_wait": probs[1],
        "predicted_action_prob_skip": probs[2],
    }


def test_three_action_label_assigns_buy_wait_skip():
    frame = pd.DataFrame(
        [
            _row(0, base10=3.0),
            _row(5, base10=1.0),
            _row(10, base10=4.0),
            _row(15, base10=-2.0),
            _row(20, base10=-1.0),
        ]
    )
    labels = three_action_label(frame)

    assert labels.iloc[0] == ACTION_BUY
    assert labels.iloc[1] == ACTION_WAIT
    assert labels.iloc[2] == ACTION_BUY
    assert labels.iloc[3] == ACTION_SKIP
    assert pd.isna(labels.iloc[4])


def test_buy_wait_skip_policy_waits_then_buys():
    frame = pd.DataFrame(
        [
            _row(0, base10=-1.0, action=ACTION_WAIT),
            _row(5, base10=2.0, action=ACTION_BUY),
            _row(10, base10=3.0, action=ACTION_BUY),
        ]
    )
    trades, paths = select_policy(frame, policy="buy_wait_skip_cap1")

    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 5 * MINUTE_MS
    assert paths["buy_at_5"] == 1


def test_buy_wait_skip_policy_predicted_skip_terminates():
    frame = pd.DataFrame(
        [
            _row(0, base10=-1.0, action=ACTION_SKIP),
            _row(5, base10=10.0, action=ACTION_BUY),
        ]
    )
    trades, paths = select_policy(frame, policy="buy_wait_skip_cap1")

    assert trades.empty
    assert paths["skip_predicted"] == 1


def test_buy_wait_skip_wait_at_final_checkpoint_skips():
    frame = pd.DataFrame(
        [
            _row(0, base10=-1.0, action=ACTION_WAIT),
            _row(5, base10=-1.0, action=ACTION_WAIT),
            _row(10, base10=10.0, action=ACTION_WAIT),
        ]
    )
    trades, paths = select_policy(frame, policy="buy_wait_skip_cap1")

    assert trades.empty
    assert paths["skip_wait_end"] == 1


def test_buy_wait_skip_missing_checkpoint_terminates():
    frame = pd.DataFrame(
        [
            _row(0, base10=-1.0, action=ACTION_WAIT),
            _row(6, base10=10.0, action=ACTION_BUY),
        ]
    )
    trades, paths = select_policy(frame, policy="buy_wait_skip_cap1")

    assert trades.empty
    assert paths["skip_missing_checkpoint"] == 1


def test_buy_wait_skip_success_check_requires_all_rules():
    rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        rows.extend(
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
                    "policy": "buy_wait_skip_cap1",
                    "trades": 20,
                    "base_mean_pct": 1.0,
                    "day_balanced_base_mean_pct": 1.0,
                    "base_p05_pct": -9.0,
                    "stress_mean_pct": 0.0,
                },
            ]
        )
    details = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02", "2026-03"],
            "balanced_accuracy": [0.34, 0.35, 0.36],
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

    diagnostics.loc[1, "balanced_accuracy"] = 0.32
    assert success_check(
        details,
        diagnostics,
        coverage,
        {"ci_low_pct": 0.1},
    )["all_pass"] is False
