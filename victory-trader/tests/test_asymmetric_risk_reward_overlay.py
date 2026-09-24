import math

import pandas as pd

import victory_trader.asymmetric_risk_reward_overlay as target


def _episode(marks, exits):
    rows = []
    for minute, (mark, exit_now) in enumerate(
        zip(marks, exits),
        start=1,
    ):
        rows.append(
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1_000_000,
                "minutes_held": float(minute),
                "test_mark": float(mark),
                "exit_now_base_return_pct": float(exit_now),
                "next_minute_base_return_pct": float(exit_now),
            }
        )
    return pd.DataFrame(rows)


def test_hard_stop_uses_realized_executable_exit(monkeypatch):
    monkeypatch.setattr(
        target,
        "_marked_base_return",
        lambda row: float(row["test_mark"]),
    )
    frame = _episode([-6.0, -2.0], [-7.25, -1.0])
    spec = target.PolicySpec(-5.0, 10.0, 5.0)
    trajectories, _ = target.build_policy_trajectories(
        frame,
        spec,
    )
    row = trajectories.iloc[0]
    assert row["exit_reason"] == "hard_stop"
    assert row["minutes_held"] == 1.0
    assert row["base_net_return_pct"] == -7.25
    assert not bool(row["partial_taken"])


def test_half_take_then_trailing_exit(monkeypatch):
    monkeypatch.setattr(
        target,
        "_marked_base_return",
        lambda row: float(row["test_mark"]),
    )
    frame = _episode(
        [6.0, 12.0, 7.0],
        [4.0, 10.0, 6.0],
    )
    spec = target.PolicySpec(-5.0, 5.0, 5.0)
    trajectories, _ = target.build_policy_trajectories(
        frame,
        spec,
    )
    row = trajectories.iloc[0]
    assert bool(row["partial_taken"])
    assert row["partial_minute"] == 1.0
    assert row["exit_reason"] == "trailing_exit"
    assert row["minutes_held"] == 3.0
    assert row["base_net_return_pct"] == 5.0


def test_reference_account_releases_partial_cash():
    trajectories = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1_000_000,
                "status": "completed",
                "base_net_return_pct": 10.0,
                "minutes_held": 3.0,
                "partial_taken": True,
                "partial_return_pct": 5.0,
                "partial_minute": 1.0,
                "partial_fraction": 0.5,
                "final_return_pct": 15.0,
                "final_fraction": 0.5,
            }
        ]
    )
    account = target.simulate_reference_account(
        trajectories
    )
    assert account["admitted_trades"] == 1
    assert account["winning_trades"] == 1
    assert math.isclose(
        account["ending_balance_krw"],
        1_020_000.0,
        rel_tol=0,
        abs_tol=1e-8,
    )


def test_silent_minute_does_not_force_exit(monkeypatch):
    monkeypatch.setattr(
        target,
        "_marked_base_return",
        lambda row: float(row["test_mark"]),
    )
    frame = _episode(
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
    )
    frame.loc[1, "minutes_held"] = 3.0
    frame.loc[2, "minutes_held"] = 4.0
    spec = target.PolicySpec(-5.0, 10.0, 5.0)
    trajectories, _ = target.build_policy_trajectories(
        frame,
        spec,
    )
    row = trajectories.iloc[0]
    assert row["exit_reason"] != "missing_state_next_minute_exit"
    assert row["exit_reason"] != "missing_state_delayed_open_exit"
    assert row["minutes_held"] != 2.0


def test_exact_scan_open_resolves_forced_cap(monkeypatch):
    monkeypatch.setattr(
        target,
        "_marked_base_return",
        lambda row: float(row["test_mark"]),
    )
    frame = _episode([0.0], [0.0])
    frame["entry_open"] = 100.0
    hot_t = int(frame.iloc[0]["hot_t"])
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "t": hot_t
                + 31 * target.MINUTE_MS,
                "o": 110.0,
            }
        ]
    )
    trajectories, _ = target.build_policy_trajectories(
        frame,
        target.PolicySpec(None, None, None),
        scan,
    )
    row = trajectories.iloc[0]
    assert row["status"] == "completed"
    assert row["exit_reason"] == "forced_30m_cap_exact_scan"
    assert row["minutes_held"] == 30.0
    assert math.isclose(
        row["base_net_return_pct"],
        target._base_return(100.0, 110.0),
        rel_tol=0,
        abs_tol=1e-12,
    )
