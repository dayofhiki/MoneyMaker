import numpy as np

from victory_trader import causal_pullback_entry_audit as r253


def test_pullback_rules_are_causal_and_precommitted():
    events = [
        (60_000, 10.0),
        (120_000, 10.5),
        (180_000, 10.3),
        (240_000, 10.2),
        (300_000, 10.35),
    ]
    assert r253.pick_entry(events, "immediate") == events[0]
    assert r253.pick_entry(events, "pullback_1") == events[2]
    assert r253.pick_entry(events, "pullback_2") == events[3]
    assert r253.pick_entry(events, "pullback_3") is None
    assert r253.pick_entry(events, "rebound_2_1") == events[4]


def test_fixed_value_after_entry_uses_precommitted_horizons():
    episode = [
        (60_000, 10.0),
        (120_000, 10.1),
        (180_000, 10.2),
        (240_000, 10.3),
        (300_000, 10.4),
        (360_000, 10.5),
    ]
    value = r253.fixed_value_after_entry(
        episode,
        hot_t=0,
        entry=episode[0],
    )
    assert np.isfinite(value)


def test_request253_contract():
    assert r253.REQUEST_ID == 253
    assert r253.ENTRY_WINDOW_MINUTES == 10
    assert r253.RISK_CAP_MINUTES == 30
    assert r253.EVENT_HORIZONS == (1, 2, 3, 5)
    assert r253.CANDIDATE_FRACTIONS == (0.05, 0.10, 0.20)
    assert r253.MIN_GAIN_VS_IMMEDIATE == 0.50
