from __future__ import annotations

import pytest

from victory_trader.attention_runtime import (
    AttentionConfig,
    AttentionEvidence,
    AttentionRuntime,
    AttentionState,
    ObservationResolution,
)


def _evidence(
    ticker: str,
    score: float,
    *,
    position_open: bool = False,
    position_just_closed: bool = False,
    data_valid: bool = True,
    eligible: bool = True,
) -> AttentionEvidence:
    return AttentionEvidence(
        ticker=ticker,
        attention_score=score,
        position_open=position_open,
        position_just_closed=position_just_closed,
        data_valid=data_valid,
        eligible=eligible,
    )


def _runtime(**kwargs: object) -> AttentionRuntime:
    values = {
        "watch_enter_score": 0.50,
        "watch_exit_score": 0.30,
        "hot_enter_score": 0.80,
        "hot_exit_score": 0.60,
        "max_watch": 2,
        "max_hot": 1,
        "drop_after_missed_batches": 2,
    }
    values.update(kwargs)
    return AttentionRuntime(AttentionConfig(**values))


def test_market_wide_batch_keeps_low_score_symbols_in_scan():
    plans = _runtime().step(
        60_000,
        [_evidence("AAA", 0.10), _evidence("BBB", 0.20)],
    )

    assert set(plans) == {"AAA", "BBB"}
    assert {plan.state for plan in plans.values()} == {AttentionState.SCAN}
    assert {
        plan.resolution for plan in plans.values()
    } == {ObservationResolution.GROUPED_MINUTE}


def test_attention_budget_promotes_ranked_names_and_reallocates_to_new_leader():
    runtime = _runtime()
    first = runtime.step(
        60_000,
        [
            _evidence("AAA", 0.90),
            _evidence("BBB", 0.70),
            _evidence("CCC", 0.60),
        ],
    )

    assert first["AAA"].state is AttentionState.HOT
    assert first["AAA"].resolution is ObservationResolution.SECOND_BARS
    assert first["BBB"].state is AttentionState.WATCH
    assert first["CCC"].state is AttentionState.WATCH

    second = runtime.step(
        120_000,
        [
            _evidence("AAA", 0.65),
            _evidence("BBB", 0.70),
            _evidence("CCC", 0.60),
            _evidence("DDD", 0.95),
        ],
    )

    assert second["DDD"].state is AttentionState.HOT
    assert second["AAA"].state is AttentionState.WATCH
    assert second["BBB"].state is AttentionState.WATCH
    assert second["CCC"].state is AttentionState.SCAN


def test_hysteresis_prevents_watch_and_hot_threshold_chatter():
    runtime = _runtime()
    runtime.step(60_000, [_evidence("AAA", 0.90)])

    still_hot = runtime.step(120_000, [_evidence("AAA", 0.65)])
    assert still_hot["AAA"].state is AttentionState.HOT

    still_watched = runtime.step(180_000, [_evidence("AAA", 0.45)])
    assert still_watched["AAA"].state is AttentionState.WATCH

    cooled = runtime.step(240_000, [_evidence("AAA", 0.20)])
    assert cooled["AAA"].state is AttentionState.SCAN


def test_open_position_is_pinned_and_closed_position_returns_to_watch():
    runtime = _runtime(max_hot=0, max_watch=1)
    open_plans = runtime.step(
        60_000,
        [
            _evidence("AAA", 0.10, position_open=True),
            _evidence("BBB", 0.90),
        ],
    )

    assert open_plans["AAA"].state is AttentionState.POSITION
    assert open_plans["AAA"].resolution is ObservationResolution.TRADES_NBBO
    assert open_plans["BBB"].state is AttentionState.WATCH

    closed_plans = runtime.step(
        120_000,
        [
            _evidence("AAA", 0.10, position_just_closed=True),
            _evidence("BBB", 0.90),
        ],
    )

    assert closed_plans["AAA"].state is AttentionState.WATCH
    assert closed_plans["BBB"].state is AttentionState.SCAN


def test_invalid_symbol_drops_but_can_reenter_on_later_causal_evidence():
    runtime = _runtime()
    invalid = runtime.step(
        60_000,
        [_evidence("AAA", 0.95, data_valid=False)],
    )
    assert invalid["AAA"].state is AttentionState.DROP
    assert invalid["AAA"].resolution is ObservationResolution.NONE

    recovered = runtime.step(120_000, [_evidence("AAA", 0.95)])
    assert recovered["AAA"].state is AttentionState.HOT


def test_missing_symbols_age_out_while_missing_positions_remain_pinned():
    runtime = _runtime()
    runtime.step(
        60_000,
        [
            _evidence("AAA", 0.70),
            _evidence("POS", 0.10, position_open=True),
        ],
    )

    once_missing = runtime.step(120_000, [])
    assert once_missing["AAA"].state is AttentionState.WATCH
    assert once_missing["POS"].state is AttentionState.POSITION

    twice_missing = runtime.step(180_000, [])
    assert twice_missing["AAA"].state is AttentionState.DROP
    assert twice_missing["POS"].state is AttentionState.POSITION
    assert twice_missing["POS"].reason == "position_feed_missing"


def test_missing_grace_states_reserve_capacity_until_they_drop():
    runtime = _runtime(max_watch=1, max_hot=1, drop_after_missed_batches=3)
    first = runtime.step(
        60_000,
        [_evidence("HOT", 0.90), _evidence("WATCH", 0.70)],
    )
    assert first["HOT"].state is AttentionState.HOT
    assert first["WATCH"].state is AttentionState.WATCH

    second = runtime.step(
        120_000,
        [_evidence("NEW_HOT", 0.95), _evidence("NEW_WATCH", 0.75)],
    )
    assert second["HOT"].state is AttentionState.HOT
    assert second["WATCH"].state is AttentionState.WATCH
    assert second["NEW_HOT"].state is AttentionState.SCAN
    assert second["NEW_WATCH"].state is AttentionState.SCAN

    runtime.step(
        180_000,
        [_evidence("NEW_HOT", 0.95), _evidence("NEW_WATCH", 0.75)],
    )
    fourth = runtime.step(
        240_000,
        [_evidence("NEW_HOT", 0.95), _evidence("NEW_WATCH", 0.75)],
    )
    assert fourth["HOT"].state is AttentionState.DROP
    assert fourth["WATCH"].state is AttentionState.DROP
    assert fourth["NEW_HOT"].state is AttentionState.HOT
    assert fourth["NEW_WATCH"].state is AttentionState.WATCH


def test_runtime_rejects_duplicate_tickers_and_time_regression():
    runtime = _runtime()
    runtime.step(60_000, [_evidence("AAA", 0.50)])

    with pytest.raises(ValueError, match="strictly increase"):
        runtime.step(60_000, [_evidence("AAA", 0.60)])

    with pytest.raises(ValueError, match="duplicate ticker"):
        runtime.step(
            120_000,
            [_evidence("AAA", 0.60), _evidence("AAA", 0.70)],
        )


def test_snapshot_is_serializable_and_restorable_without_changing_state():
    runtime = _runtime()
    runtime.step(
        60_000,
        [_evidence("AAA", 0.90), _evidence("BBB", 0.60)],
    )

    payload = runtime.snapshot()
    restored = AttentionRuntime.from_snapshot(runtime.config, payload)

    assert restored.snapshot() == payload
    next_plans = restored.step(
        120_000,
        [_evidence("AAA", 0.65), _evidence("BBB", 0.40)],
    )
    assert next_plans["AAA"].state is AttentionState.HOT
    assert next_plans["BBB"].state is AttentionState.WATCH
