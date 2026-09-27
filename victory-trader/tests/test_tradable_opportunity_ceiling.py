import pandas as pd

from victory_trader import tradable_opportunity_ceiling as r248


def test_frozen_exit_time_uses_first_nonpositive_hold_prediction():
    part = pd.DataFrame(
        [
            {"trading_day":"2026-05-11","ticker":"TEST","hot_t":0,"state_t":60_000,"terminal":False},
            {"trading_day":"2026-05-11","ticker":"TEST","hot_t":0,"state_t":120_000,"terminal":False},
            {"trading_day":"2026-05-11","ticker":"TEST","hot_t":0,"state_t":180_000,"terminal":False},
        ]
    )
    predicted = pd.Series([0.0, 0.5, -0.2], index=part.index)
    assert r248.frozen_exit_time(part, 0, predicted) == 180_000


def test_opportunity_ceiling_oracle_is_episode_diagnostic_only():
    rows = pd.DataFrame(
        [
            {"trading_day":"2026-05-11","ticker":"A","hot_t":1,"state_t":1,"forced_base_return":-1.0},
            {"trading_day":"2026-05-11","ticker":"A","hot_t":1,"state_t":2,"forced_base_return":2.0},
            {"trading_day":"2026-05-11","ticker":"B","hot_t":2,"state_t":1,"forced_base_return":-0.5},
        ]
    )
    episodes = pd.DataFrame(
        [
            {"trading_day":"2026-05-11","ticker":"A","hot_t":1},
            {"trading_day":"2026-05-11","ticker":"B","hot_t":2},
            {"trading_day":"2026-05-11","ticker":"C","hot_t":3},
        ]
    )
    report = r248.opportunity_ceiling(rows, episodes)
    assert report["episodes_with_admitted_state"] == 2
    assert report["episodes_with_resolved_admitted_state"] == 2
    assert report["positive_opportunity_rate_among_resolved_admitted_episodes"] == 0.5
    assert report["diagnostic_oracle_best_return_mean_pct"] == 0.75


def test_diagnose_candidate_population_when_oracle_ceiling_is_weak():
    ceiling = {
        "positive_opportunity_rate_among_resolved_admitted_episodes": 0.2,
        "diagnostic_oracle_best_return_mean_pct": 1.0,
    }
    assert r248.diagnose(ceiling, {"positive_auc":0.8,"spearman":0.3}) == (
        "candidate_population_bottleneck"
    )


def test_diagnose_representation_when_ceiling_exists_but_ranking_is_weak():
    ceiling = {
        "positive_opportunity_rate_among_resolved_admitted_episodes": 0.5,
        "diagnostic_oracle_best_return_mean_pct": 2.0,
    }
    assert r248.diagnose(ceiling, {"positive_auc":0.55,"spearman":0.05}) == (
        "economic_representation_bottleneck"
    )


def test_request248_contract_is_frozen():
    assert r248.REQUEST_ID == 248
    assert r248.LEARN_DAYS == [
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
        "2026-05-08",
    ]
    assert r248.TEST_DAYS == [
        "2026-05-11",
        "2026-05-12",
        "2026-05-13",
        "2026-05-14",
        "2026-05-15",
        "2026-05-18",
        "2026-05-19",
        "2026-05-20",
    ]
    assert r248.ADMISSION_THRESHOLD == 0.60
