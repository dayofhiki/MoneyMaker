import pandas as pd

from victory_trader.post_pullback_recheck_learnability import (
    build_recheck_states,
)


def test_recheck_states_are_fixed_offsets_after_first_pullback(monkeypatch):
    import victory_trader.post_pullback_recheck_learnability as module

    states = pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 0,
            "state_t": 60_000,
            "state_price": 10.0,
            "drawdown_from_running_high_pct": 0.0,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 0,
            "state_t": 120_000,
            "state_price": 9.7,
            "drawdown_from_running_high_pct": -3.0,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 0,
            "state_t": 180_000,
            "state_price": 9.8,
            "drawdown_from_running_high_pct": -2.0,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 0,
            "state_t": 240_000,
            "state_price": 9.9,
            "drawdown_from_running_high_pct": -1.0,
        },
    ])

    monkeypatch.setattr(
        module,
        "build_episode_states",
        lambda candidates, scan: states,
    )
    out = build_recheck_states(
        pd.DataFrame([{
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "t": 0,
        }]),
        pd.DataFrame(),
    )
    assert out.recheck_horizon_events.tolist() == [0, 1, 2]
    assert out.entry_t.tolist() == [120_000, 180_000, 240_000]
