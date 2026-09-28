import numpy as np
import pandas as pd

from victory_trader import pullback_trailing_exit as r273


def test_trailing_stop_uses_running_post_entry_high():
    episodes = pd.DataFrame(
        [{
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "hot_t": 0,
            "entered": True,
            "entry_t": 60_000,
            "entry_price": 10.0,
            "path": [
                (120_000, 10.5),
                (180_000, 10.4),
                (240_000, 10.2),
            ],
        }]
    )
    rows = r273.apply_exit_rule(
        episodes,
        stop_loss_pct=-4.0,
        trail_pct=2.0,
        max_hold_minutes=30,
        policy_name="test",
    )
    assert rows.loc[0, "exit_t"] == 240_000
    assert rows.loc[0, "exit_reason"] == "trailing_stop"
    assert np.isclose(
        rows.loc[0, "trade_return_pct"],
        r273._base_return(10.0, 10.2),
    )


def test_hard_stop_precedes_later_recovery():
    episodes = pd.DataFrame(
        [{
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "hot_t": 0,
            "entered": True,
            "entry_t": 60_000,
            "entry_price": 10.0,
            "path": [
                (120_000, 9.7),
                (180_000, 10.8),
            ],
        }]
    )
    rows = r273.apply_exit_rule(
        episodes,
        stop_loss_pct=-2.5,
        trail_pct=3.0,
        max_hold_minutes=30,
        policy_name="test",
    )
    assert rows.loc[0, "exit_t"] == 120_000
    assert rows.loc[0, "exit_reason"] == "hard_stop"
    assert np.isclose(
        rows.loc[0, "trade_return_pct"],
        r273._base_return(10.0, 9.7),
    )


def test_cash_episode_is_resolved_zero():
    episodes = pd.DataFrame(
        [{
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "hot_t": 0,
            "entered": False,
            "entry_t": None,
            "entry_price": None,
            "path": [],
        }]
    )
    rows = r273.apply_exit_rule(
        episodes,
        stop_loss_pct=-2.5,
        trail_pct=2.0,
        max_hold_minutes=10,
        policy_name="test",
    )
    assert bool(rows.loc[0, "resolved"])
    assert rows.loc[0, "economic_return_pct"] == 0.0
    assert pd.isna(rows.loc[0, "trade_return_pct"])


def test_request273_contract():
    assert r273.REQUEST_ID == 273
    assert r273.RULE_FIT_DAYS == (
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
    )
    assert r273.RULE_CHECK_DAY == "2026-05-08"
    assert r273.STOP_LOSSES == (-1.5, -2.5, -4.0)
    assert r273.TRAILS == (1.0, 2.0, 3.0)
    assert r273.MAX_HOLD_MINUTES == (5, 10, 20, 30)
