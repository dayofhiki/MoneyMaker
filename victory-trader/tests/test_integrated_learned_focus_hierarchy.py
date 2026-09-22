from __future__ import annotations

import pandas as pd

from victory_trader.integrated_learned_focus_hierarchy import (
    EVAL_DAYS,
    add_learned_focus_state,
    evaluate_integration,
)


def test_learned_focus_state_tracks_only_consecutive_membership():
    focus = pd.DataFrame(
        [
            {
                "trading_day": "2026-03-04",
                "ticker": "AAA",
                "t": t,
                "market_hazard_probability": 0.9,
            }
            for t in [60_000, 120_000, 240_000]
        ]
    )

    result = add_learned_focus_state(focus)

    assert result["focus_age_minutes"].tolist() == [1, 2, 1]
    assert result["watch_run_age_minutes"].tolist() == [1, 2, 1]
    assert result["stage1_hazard_probability"].tolist() == [0.9, 0.9, 0.9]


def _integration_frames() -> tuple[pd.DataFrame, ...]:
    crossings: list[dict[str, object]] = []
    frozen: list[dict[str, object]] = []
    integrated: list[dict[str, object]] = []
    population: list[dict[str, object]] = []
    for day in EVAL_DAYS:
        for index in range(10):
            ticker = f"T{index:02d}"
            crossings.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "t": 180_000,
                    "runner_cross_now": True,
                }
            )
            population.append(
                {"trading_day": day, "ticker": ticker, "t": 120_000}
            )
            integrated.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "t": 120_000,
                    "state": "hot",
                }
            )
            if index < 4:
                frozen.append(
                    {
                        "trading_day": day,
                        "ticker": ticker,
                        "t": 120_000,
                        "state": "hot",
                    }
                )
    return tuple(
        pd.DataFrame(rows)
        for rows in [crossings, frozen, integrated, population]
    )


def test_integration_gate_passes_when_full_handoff_improves():
    scan, frozen, integrated, population = _integration_frames()

    result = evaluate_integration(
        scan,
        frozen,
        integrated,
        population,
        population,
        1.0,
        {"max_hot_occupancy": 10},
    )

    assert result["frozen_hierarchy"]["hot_within_1m_rate"] == 0.4
    assert result["integrated_hierarchy"]["hot_within_1m_rate"] == 1.0
    assert result["nonlower_immediate_capture_days"] == 5
    assert result["promotion_gate_pass"] is True


def test_integration_gate_rejects_nonimproving_handoff():
    scan, frozen, _, population = _integration_frames()

    result = evaluate_integration(
        scan,
        frozen,
        frozen,
        population,
        population,
        1.0,
        {"max_hot_occupancy": 10},
    )

    assert result["promotion_gate_pass"] is False
