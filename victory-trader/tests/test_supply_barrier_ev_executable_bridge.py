import pandas as pd

from victory_trader.supply_barrier_ev_executable_bridge import (
    barrier_summary,
    select_positive_ev,
)


def _frame():
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "AAA",
                "hot_t": 1,
                "state_t": 100,
                "minutes_held": 1.0,
                "supply_barrier_ev": 1.0,
                "first_passage_status": "entry_unavailable",
                "barrier_proxy_pct": None,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "AAA",
                "hot_t": 1,
                "state_t": 200,
                "minutes_held": 2.0,
                "supply_barrier_ev": 0.5,
                "first_passage_status": "take_first",
                "barrier_proxy_pct": 10.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "BBB",
                "hot_t": 2,
                "state_t": 100,
                "minutes_held": 1.0,
                "supply_barrier_ev": -0.1,
                "first_passage_status": "stop_first",
                "barrier_proxy_pct": -3.0,
            },
        ]
    )


def test_positive_ev_retries_after_unavailable_entry():
    selected, audit = select_positive_ev(
        _frame(),
        "supply_barrier_ev",
    )
    assert len(selected) == 1
    assert selected.iloc[0]["ticker"] == "AAA"
    assert selected.iloc[0]["minutes_held"] == 2.0
    assert audit["signaled_episodes"] == 1
    assert audit["expired_entry_attempts"] == 1
    assert audit["admitted_episodes"] == 1


def test_barrier_summary_uses_economic_proxy():
    selected, _ = select_positive_ev(
        _frame(),
        "supply_barrier_ev",
    )
    summary = barrier_summary(selected)
    assert summary["evaluable_coverage"] == 1.0
    assert summary["mean_proxy_pct"] == 10.0
    assert summary["day_balanced_proxy_pct"] == 10.0
    assert summary["positive_days"] == 1
    assert summary["take_first_rate"] == 1.0
