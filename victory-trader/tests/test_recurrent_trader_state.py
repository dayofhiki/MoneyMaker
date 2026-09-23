from __future__ import annotations

import pytest

from victory_trader.recurrent_trader_state import CausalEpisodeMemory, MINUTE_MS


def test_episode_memory_survives_hot_demotion_and_repromotion():
    memory = CausalEpisodeMemory()
    day = "2026-05-21"

    first = memory.observe(
        trading_day=day,
        ticker="abc",
        t=10 * MINUTE_MS,
        state="hot",
        hot_score=0.20,
        opportunity_score=0.60,
    )
    assert first.promotion_index == 1
    assert first.minutes_since_previous_promotion is None
    assert first.hot_score_peak == pytest.approx(0.20)

    cooled = memory.observe(
        trading_day=day,
        ticker="abc",
        t=14 * MINUTE_MS,
        state="watch",
        hot_score=0.10,
        opportunity_score=0.45,
    )
    assert cooled.promotion_index == 1
    assert cooled.minutes_continuously_hot == 0.0
    assert cooled.hot_score_peak == pytest.approx(0.20)
    assert cooled.hot_score_drawdown_from_peak == pytest.approx(-0.10)

    second = memory.observe(
        trading_day=day,
        ticker="abc",
        t=17 * MINUTE_MS,
        state="hot",
        hot_score=0.18,
        opportunity_score=0.55,
    )
    assert second.promotion_index == 2
    assert second.minutes_since_previous_promotion == pytest.approx(7.0)
    assert second.hot_score_peak == pytest.approx(0.20)
    assert second.opportunity_score_peak == pytest.approx(0.60)


def test_episode_memory_tracks_continuous_hot_and_observation_age():
    memory = CausalEpisodeMemory()
    first = memory.observe(
        trading_day="2026-05-21",
        ticker="A",
        t=100 * MINUTE_MS,
        state="hot",
    )
    later = memory.observe(
        trading_day="2026-05-21",
        ticker="A",
        t=103 * MINUTE_MS,
        state="hot",
    )
    assert first.minutes_continuously_hot == 0.0
    assert later.minutes_continuously_hot == pytest.approx(3.0)
    assert later.minutes_continuously_observed == pytest.approx(3.0)

    memory.clear_observation(trading_day="2026-05-21", ticker="A")
    resumed = memory.observe(
        trading_day="2026-05-21",
        ticker="A",
        t=105 * MINUTE_MS,
        state="watch",
    )
    assert resumed.minutes_continuously_observed == 0.0


def test_episode_memory_tracks_position_and_resets_next_session():
    memory = CausalEpisodeMemory()
    memory.observe(
        trading_day="2026-05-21",
        ticker="A",
        t=MINUTE_MS,
        state="hot",
        position_open=True,
    )
    still_open = memory.observe(
        trading_day="2026-05-21",
        ticker="A",
        t=2 * MINUTE_MS,
        state="hot",
    )
    assert still_open.position_open is True

    next_day = memory.observe(
        trading_day="2026-05-22",
        ticker="A",
        t=MINUTE_MS,
        state="hot",
    )
    assert next_day.promotion_index == 1
    assert next_day.position_open is False


def test_episode_memory_rejects_backward_time():
    memory = CausalEpisodeMemory()
    memory.observe(
        trading_day="2026-05-21",
        ticker="A",
        t=10,
        state="watch",
    )
    with pytest.raises(ValueError):
        memory.observe(
            trading_day="2026-05-21",
            ticker="A",
            t=9,
            state="watch",
        )
