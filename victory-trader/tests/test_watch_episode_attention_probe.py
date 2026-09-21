from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.second_path_attention_probe import (
    BASELINE_FEATURES,
    SECOND_FEATURES,
)
from victory_trader.watch_episode_attention_probe import (
    _episode_target,
    enumerate_watch_episodes,
    fit_and_evaluate_episode,
)


def test_watch_episode_enumeration_skips_direct_hot_episode():
    trace = pd.DataFrame(
        [
            {"ticker": "AAA", "t": 0, "state": "scan"},
            {"ticker": "AAA", "t": 60_000, "state": "watch"},
            {"ticker": "AAA", "t": 120_000, "state": "hot"},
            {"ticker": "AAA", "t": 180_000, "state": "watch"},
            {"ticker": "AAA", "t": 240_000, "state": "scan"},
            {"ticker": "BBB", "t": 0, "state": "scan"},
            {"ticker": "BBB", "t": 60_000, "state": "hot"},
            {"ticker": "BBB", "t": 120_000, "state": "watch"},
            {"ticker": "BBB", "t": 180_000, "state": "scan"},
            {"ticker": "CCC", "t": 0, "state": "scan"},
            {"ticker": "CCC", "t": 60_000, "state": "watch"},
            {"ticker": "CCC", "t": 120_000, "state": "watch"},
        ]
    )

    episodes = enumerate_watch_episodes(trace)

    assert [(e.ticker, e.entry_t, e.end_t, e.end_exclusive) for e in episodes] == [
        ("AAA", 60_000, 240_000, True),
        ("CCC", 60_000, 120_000, False),
    ]


def test_episode_target_excludes_nonfocus_terminal_decision():
    scan = pd.DataFrame(
        [
            {"t": 60_000, "c": 10.0, "runner_cross_now": False},
            {"t": 120_000, "c": 11.0, "runner_cross_now": False},
            {"t": 180_000, "c": 20.0, "runner_cross_now": True},
        ]
    )

    peak, runner, points = _episode_target(
        scan,
        entry_t=60_000,
        end_t=180_000,
        end_exclusive=True,
        entry_close=10.0,
    )

    assert peak == 10.0
    assert runner is False
    assert points == 1


def test_episode_model_uses_fresh_january_evaluation_days():
    rng = np.random.default_rng(19)
    days = [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
        "2026-01-16",
    ]
    rows = []
    for day_index, day in enumerate(days):
        for i in range(20):
            baseline_signal = i / 20.0
            second_signal = ((i * 3 + day_index) % 20) / 20.0
            row = {
                "trading_day": day,
                "ticker": f"T{i:02d}",
                "remaining_episode_peak_return_pct": max(
                    0.0,
                    baseline_signal + second_signal + rng.normal(0, 0.01),
                ),
                "runner_in_remaining_episode": (i % 5 == 0),
                "episode_duration_minutes": 2.0 + (i % 6),
            }
            for feature in BASELINE_FEATURES:
                row[feature] = baseline_signal
            for feature in SECOND_FEATURES:
                row[feature] = second_signal
            rows.append(row)

    modeled, summary = fit_and_evaluate_episode(pd.DataFrame(rows))

    assert summary["fit_days"] == days[:6]
    assert summary["eval_days"] == days[6:]
    assert summary["fit_rows"] == 120
    assert summary["eval_rows"] == 100
    assert set(modeled.loc[modeled["split"].eq("eval"), "trading_day"]) == set(
        days[6:]
    )
    assert modeled["baseline_prediction"].notna().all()
    assert modeled["extended_prediction"].notna().all()
